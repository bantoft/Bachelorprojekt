import torch
from neuralop.models.base_model import BaseModel


FIELDS = ("lnn", "lnpe", "lnpi", "phi")


class WrappedFNO(BaseModel):
    def __init__(self, fno, info, phys, m: int = 3):
        super().__init__()
        self.info, self.phys, self.fno, self.m = info, phys, fno.to(info.device), m
        self.device, self.dtype = info.device, info.dtype
        self.nx = self.info.parameters.num_x
        self.lx = self.info.parameters.Lx
        self.lz = self.info.parameters.Lz
        self.lt = self.info.parameters.Lt
        self.dz = self.info.parameters.dz
        self.half = m // 2
        self.x_line = torch.linspace(0, self.lx, self.nx, device=self.device, dtype=self.dtype)
        self.z_offset = torch.arange(-self.half, self.half + 1, device=self.device, dtype=self.dtype) * self.dz

        # Register buffers
        _buffer_dict = {"x_base": torch.linspace(0, self.lx, self.nx, device=self.device, dtype=self.dtype)[None, :, None].expand(1, self.nx, self.m).clone(),
                            "g_x": self._expand(torch.stack([phys.ic[f"init_{field}"] for field in FIELDS])),
                            "mean": self.info.standardized.mean.to(self.device, self.dtype).view(1, -1, 1, 1),
                            "std": self.info.standardized.std.to(self.device, self.dtype).view(1, -1, 1, 1)}
            
        for name, value in _buffer_dict.items():
            self.register_buffer(name, value)


        h_x, d_x = zip(*(self._get_Hx_Dx(field) for field in FIELDS))
        self.register_buffer("H_x", self._expand(torch.stack(h_x)))
        self.register_buffer("D_x", self._expand(torch.stack(d_x)))

    def _expand(self, tensor):
        return tensor[:, :, None].expand(-1, -1, self.m).to(self.device, self.dtype).clone()

    def _get_Hx_Dx(self, field_name):
        bc, x = self.phys.bc[field_name], self.x_line
        s = (x - x[0]) / (x[-1] - x[0])
        powers = torch.arange(len(bc), device=self.device, dtype=self.dtype)
        basis, rows, rhs = s[:, None] ** powers, [], []
        p_left = p_right = 0

        for side, (bc_type, value) in bc.items():
            s0 = torch.tensor(float(side != "xin"), device=self.device, dtype=self.dtype)
            if "neumann" in bc_type.lower():
                row = torch.zeros_like(powers)
                row[1:] = powers[1:] * s0 ** (powers[1:] - 1)
                value = (x[-1] - x[0]) * value
                p = 2
            else:
                row, p = s0 ** powers, 1
            rows.append(row)
            rhs.append(torch.as_tensor(value, device=self.device, dtype=self.dtype))
            p_left, p_right = (p, p_right) if side == "xin" else (p_left, p)

        H_x = basis @ torch.linalg.solve(torch.stack(rows), torch.stack(rhs))
        d_x =  (s**p_left * (1 - s)**p_right)**2
        return H_x, d_x

    def _create_model_input(self, u, center, x_base=None, requires_coord_grad: bool = True):

        u_grouped = u.reshape(u.shape[0], -1, self.mean.shape[1], *u.shape[2:])
        u_std = ((u_grouped - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)).reshape_as(u)

        num_rows = x_base.shape[1]

        z_line = center[:, 0:1] + self.z_offset
        # Create target coordinate grid for x, z og t
        coord = torch.stack([x_base.expand(center.shape[0], num_rows, self.m),
                             z_line[:, None, :].expand(center.shape[0], num_rows, self.m),
                             center[:, 1:2, None].expand(center.shape[0], num_rows, self.m)], dim=1)
        if requires_coord_grad:
            coord.requires_grad_(True)

        x_embedded = 2 * coord[:, 0:1] / self.lx - 1
        z1 = torch.cos(2 * torch.pi * coord[:, 1:2] / self.lz)
        z2 = torch.sin(2 * torch.pi * coord[:, 1:2] / self.lz)
        t_embedded = coord[:, 2:3] / self.lt

        model_input = torch.cat([u_std, x_embedded, z1, z2, t_embedded], dim=1)
        return model_input, coord

    def forward(self, batch, chunking=None, requires_coord_grad: bool = True):
        u = batch[0]
        center = batch[3]

        if chunking:
            row_indices = batch[4][0, 0, :, 0].long()

            row_start = int(row_indices[0].item())
            num_rows = u.shape[2]
            expected = torch.arange(row_start, row_start + num_rows, device=row_indices.device)

            if row_indices.shape[0] == num_rows and torch.equal(row_indices, expected):
                x_base = self.x_base[:, row_start : row_start + num_rows]
                g_x = self.g_x[:, row_start : row_start + num_rows]
                H_x = self.H_x[:, row_start : row_start + num_rows]
                D_x = self.D_x[:, row_start : row_start + num_rows]
            else:
                x_base = self.x_base.index_select(1, row_indices)
                g_x = self.g_x.index_select(1, row_indices)
                H_x = self.H_x.index_select(1, row_indices)
                D_x = self.D_x.index_select(1, row_indices)
        else:
            x_base = self.x_base
            g_x = self.g_x
            H_x = self.H_x
            D_x = self.D_x

        model_input, coord_fys = self._create_model_input(
            u,
            center,
            x_base=x_base,
            requires_coord_grad=requires_coord_grad,
        )

        sigma_t = 1.0 - torch.exp(-coord_fys[:, 2:3] / 25)

        N_theta = self.fno(model_input)* self.std + self.mean

        f_theta = g_x[None] + sigma_t * (H_x[None] - g_x[None] + D_x[None] * (N_theta - H_x[None]))
        f_theta = {"lnn": f_theta[:, 0:1], "lnpe": f_theta[:, 1:2], "lnpi": f_theta[:, 2:3], "phi": f_theta[:, 3:4]}

        return f_theta, coord_fys
