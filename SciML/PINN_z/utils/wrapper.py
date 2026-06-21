import torch

from SciML.PINN.util.model import PINN


FIELDS = ("lnn", "lnpe", "lnpi", "phi")


class WrappedPINN(torch.nn.Module):
    def __init__(self, network_structure, info, phys, m: int = 3):
        super().__init__()
        self.info = info
        self.phys = phys
        self.m = m
        self.device, self.dtype = info.device, info.dtype
        self.nx = self.info.parameters.num_x - 2
        self.lx = self.info.parameters.Lx
        self.lz = self.info.parameters.Lz
        self.lt = self.info.parameters.Lt
        self.dz = self.info.parameters.dz
        self.z_half = m // 2
        self.num_input_channels = 12
        self.x_line = torch.linspace(0, self.lx, self.nx, device=self.device, dtype=self.dtype)
        self.s_line = (self.x_line - self.x_line[0]) / (self.x_line[-1] - self.x_line[0])
        self.z_offset = torch.arange(-self.z_half, self.z_half + 1, device=self.device, dtype=self.dtype) * self.dz
        self.line_input_size = self.num_input_channels * self.nx * self.m
        self.line_output_size = len(FIELDS) * self.nx
        self.pinn = PINN({
            **network_structure,
            "input_size": self.line_input_size,
            "output_size": self.line_output_size,
            "output_names": tuple(network_structure["output_names"]),
        }).to(self.device)

        # Register buffers
        _buffer_dict = {"x_base": self.x_line[None, :, None].expand(1, self.nx, self.m).clone(),
                            "g_x": self._expand(torch.stack([phys.ic[f"init_{field}"][1:-1] for field in FIELDS])),
                            "mean": self.info.standardized.mean.to(self.device, self.dtype).view(1, -1, 1, 1),
                            "std": self.info.standardized.std.to(self.device, self.dtype).view(1, -1, 1, 1)}
            
        for name, value in _buffer_dict.items():
            self.register_buffer(name, value)
        # Smoothstep basis: 0 at x_min, 1 at x_max, and zero slope at both ends.
        self.register_buffer(
            "xout_neumann_basis",
            (self.s_line.square() * (3.0 - 2.0 * self.s_line))[None, None, :, None].clone(),
            persistent=False,
        )


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
        d_x =  (s**p_left * (1 - s)**p_right)
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
        center_coord = coord[:, :, :, self.z_half : self.z_half + 1]
        return model_input, center_coord

    def _hard_constraint(self, raw_field, field_name, h_field, d_field):
        bc = self.phys.bc[field_name]
        left_type = bc.get("xin", ("", None))[0].lower()
        right_type = bc.get("xout", ("", None))[0].lower()

        if field_name != "phi" and "dirichlet" in left_type and "neumann" in right_type:
            right_value = raw_field[:, :, -1:, :]
            return self.xout_neumann_basis * (right_value - h_field[:, -1:, :]) + d_field * (raw_field - h_field)

        return d_field * (raw_field - h_field)

    def forward(self, batch, requires_coord_grad: bool = True):
        u = batch[0]
        center = batch[3]
        x_base = self.x_base
        g_x = self.g_x[:, :, self.z_half : self.z_half + 1]
        H_x = self.H_x[:, :, self.z_half : self.z_half + 1]
        model_input, coord_fys = self._create_model_input(
            u,
            center,
            x_base=x_base,
            requires_coord_grad=requires_coord_grad,
        )

        sigma_t = 1.0 - torch.exp(-coord_fys[:, 2:3] / 25)

        network_input = model_input.reshape(model_input.shape[0], self.line_input_size)
        pred_std = self.pinn.network(network_input).reshape(model_input.shape[0], len(FIELDS), self.nx, 1)
        N_theta = pred_std * self.std + self.mean

        return {
            name: g_x[index : index + 1] + sigma_t * (
                H_x[index : index + 1]
                - g_x[index : index + 1]
                + self._hard_constraint(
                    N_theta[:, index : index + 1],
                    name,
                    H_x[index : index + 1],
                    self.D_x[index : index + 1, :, self.z_half : self.z_half + 1],
                )
            )
            for index, name in enumerate(self.pinn.output_names)
        }, coord_fys
