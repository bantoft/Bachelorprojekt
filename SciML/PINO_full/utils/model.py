import torch

from neuralop.models.base_model import BaseModel
from torch.utils.checkpoint import checkpoint



class WrappedFNO(BaseModel):

    def __init__(self, fno, info, phys):
        super().__init__()

        self.info = info
        self.phys = phys
        self.fno = fno

        self.dtype = self.info.dtype
        self.device = self.info.device

        self.nx = info.parameters.num_x
        self.nz = info.parameters.num_z
        self.lx = info.parameters.Lx
        self.lz = info.parameters.Lz

        self.x_line = torch.linspace(0.0, self.lx, self.nx, device=self.device, dtype=self.dtype)
        self.z_line = torch.linspace(0.0, self.lz, self.nz, device=self.device, dtype=self.dtype)
        self.time_scale = 50.0


        x_base = self.x_line[:, None].expand(self.nx, self.nz)
        z_base = self.z_line[None, :].expand(self.nx, self.nz)

        self.register_buffer("x_base", x_base[None, None])
        self.register_buffer("z_base", z_base[None, None])
        self.register_buffer("lz_tensor", torch.tensor(self.lz, device=self.device, dtype=self.dtype))

        g, H, D_x = self.make_x_constrains()
        self.register_buffer("g", g)
        self.register_buffer("H", H)
        self.register_buffer("D_x", D_x)
        self._subset_cache = {}

    def hermit_x(self, field_name):
        """
        compute hermitian polynomial h that satisfie baoundery conditions
        """
        x = self.x_line
        bc = self.phys.bc[field_name]

        alpha = x[0]
        beta = x[-1]
        L = beta - alpha

        s = (x - alpha) / L

        n = len(bc)
        powers = torch.arange(n, device=self.device, dtype=self.dtype)
        basis = s[:, None] ** powers

        M_rows = []
        c_rows = []

        for side, (bc_type, value) in bc.items():
            s0 = torch.tensor(
                0.0 if side == "xin" else 1.0,
                device=self.device,
                dtype=self.dtype,
            )

            value = torch.as_tensor(value, device=self.device, dtype=self.dtype)

            if "dirichlet" in bc_type.lower():
                row = s0 ** powers
                rhs = value

            elif "neumann" in bc_type.lower():
                row = torch.zeros_like(powers)
                row[1:] = powers[1:] * s0 ** (powers[1:] - 1)
                rhs = L * value

            else:
                raise ValueError(f"Unknown bc_type: {bc_type}")

            M_rows.append(row)
            c_rows.append(rhs)

        M = torch.stack(M_rows)
        c = torch.stack(c_rows)

        a = torch.linalg.solve(M, c)

        return basis @ a
    

    def make_x_constrains(self):
        x = self.x_line

        L_alpha = 0.5
        L_beta = 0.5

        sigma_alpha = 1.0 - torch.exp(-(x - x[0]) / L_alpha)
        sigma_beta = 1.0 - torch.exp(-(x[-1] - x) / L_beta)

        D_x = sigma_alpha**2 * sigma_beta**2
        D_x = D_x[None, :, None].expand(1, self.nx, self.nz)


        g = torch.stack([
            self.phys.ic["init_lnn"],
            self.phys.ic["init_lnpe"],
            self.phys.ic["init_lnpi"],
            self.phys.ic["init_phi"],
        ], dim=0).to(dtype=self.dtype)

        g = g[:, :, None].expand(-1, -1, self.nz)
        

        H = torch.stack([self.hermit_x("lnn"),
                         self.hermit_x("lnpe"),
                         self.hermit_x("lnpi"),
                         self.hermit_x("phi")
        ], dim=0).to(dtype=self.dtype)

        H = H[:, :, None].expand(-1, -1, self.nz)

        return g, H, D_x
    

    def _get_subset_tensors(self, row_indices):
        if row_indices is None:
            return self.x_base, self.z_base, self.g, self.H, self.D_x

        cache_key = tuple(row_indices.tolist())
        if cache_key not in self._subset_cache:
            self._subset_cache[cache_key] = (
                self.x_base.index_select(2, row_indices),
                self.z_base.index_select(2, row_indices),
                self.g.index_select(1, row_indices),
                self.H.index_select(1, row_indices),
                self.D_x.index_select(1, row_indices),
            )

        return self._subset_cache[cache_key]

    def create_model_input(self, u, t, row_indices=None):
        B, _, nx, nz = u.shape

        x_base, z_base, _, _, _ = self._get_subset_tensors(row_indices)

        x_coord = x_base.expand(B, 1, nx, nz)
        z_coord = z_base.expand(B, 1, nx, nz)
        t_coord = t.view(B, 1, 1, 1).expand(B, 1, nx, nz)

        coord = torch.cat([x_coord, z_coord, t_coord], dim=1).requires_grad_(True)

        x_coord = coord[:, 0:1]
        z_coord = coord[:, 1:2]
        t_coord = coord[:, 2:3]

        z1 = torch.sin(2.0 * torch.pi * z_coord / self.lz_tensor)
        z2 = torch.cos(2.0 * torch.pi * z_coord / self.lz_tensor)

        model_input = torch.cat([u, x_coord, z1, z2, t_coord], dim=1)

        return model_input, coord

    def forward(self, u, t, row_indices=None, use_checkpoint=None):
        if row_indices is not None:
            u = u.index_select(2, row_indices)

        if use_checkpoint is None:
            use_checkpoint = row_indices is None

        model_input, coord = self.create_model_input(u, t, row_indices=row_indices)

        if use_checkpoint:
            # Checkpointing trades compute for much lower activation memory.
            N_theta = checkpoint(self.fno, model_input, use_reentrant=False)
        else:
            N_theta = self.fno(model_input)

        sigma_t = (1.0 - torch.exp(-t / self.time_scale)).view(t.shape[0], 1, 1, 1)
        _, _, g, H, D_x = self._get_subset_tensors(row_indices)

        f_theta = g[None] + sigma_t * (H[None] - g[None] + D_x[None] * (N_theta - H[None]))

        return f_theta, coord
