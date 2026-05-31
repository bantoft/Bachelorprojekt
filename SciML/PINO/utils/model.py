import torch

from neuralop.models.base_model import BaseModel


class WrappedFNO(BaseModel):

    def __init__(self, fno, info, phys):
        super().__init__()

        self.info = info
        self.phys = phys
        self.fno = fno

        self.dtype = self.info.dtype
        self.nx = self.info.parameters.num_x
        self.nz = self.info.parameters.num_z

        self.x_line = torch.linspace(0.0, self.info.parameters.Lx, self.nx, device=info.device, dtype=self.dtype, requires_grad=True)
        self.z_line = torch.linspace(0.0, self.info.parameters.Lz, self.nz, device=info.device, dtype=self.dtype, requires_grad=True)

        grid = self.embed_coord()
        self.register_buffer("grid", grid)

        g, H, D_x = self.make_constrains()
        self.register_buffer("g", g)
        self.register_buffer("H", H)
        self.register_buffer("D_x", D_x)


    def embed_coord(self):
        """
        Enforces:
            Periodic bounderies i z-axis as extension:
            (x, z) = (x, sin(2*pi*z/Lz), cos(2*pi*z/Lz))
        
        return: fourier embedding [x, sin(2*pi*z/Lz), cos(2*pi*z/Lz)]
        """
        dtype = self.dtype
        nx = self.nx
        nz = self.nz

        x_axis = torch.linspace(0.0, self.info.parameters.Lx, nx, dtype=dtype).requires_grad_(True)
        z_axis = torch.linspace(0.0, self.info.parameters.Lz, nz, dtype=dtype).requires_grad_(True)
        x = x_axis[:, None].expand(-1, nz)

        theta = 2.0 * torch.pi * z_axis / self.info.parameters.Lz

        z1 = torch.sin(theta)[None, :].expand(nx, -1)
        z2 = torch.cos(theta)[None, :].expand(nx, -1)
        grid = torch.stack([x, z1, z2], dim=0)
        return grid


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
        powers = torch.arange(n, device=x.device, dtype=x.dtype)
        basis = s[:, None] ** powers

        M_rows = []
        c_rows = []

        for side, (bc_type, value) in bc.items():
            s0 = torch.tensor(
                0.0 if side == "xin" else 1.0,
                device=x.device,
                dtype=x.dtype,
            )

            value = torch.as_tensor(value, device=x.device, dtype=x.dtype)

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


    def make_constrains(self):
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
            self.phys.ic["init_vort"],
        ], dim=0).to(dtype=self.dtype)

        g = g[:, :, None].expand(-1, -1, self.nz)
        

        H = torch.stack([self.hermit_x("lnn"),
                         self.hermit_x("lnpe"),
                         self.hermit_x("lnpi"),
                         self.hermit_x("vort")
        ], dim=0).to(dtype=self.dtype)

        H = H[:, :, None].expand(-1, -1, self.nz)

        return g, H, D_x
    
    def forward(self, u, t):

        u = u.to(device=device, dtype=dtype)
        B = u.shape[0]

        t = torch.as_tensor(t, device=device, dtype=dtype)

        if t.ndim == 0:
            t = t.expand(B)

        # [B] -> [B, 1, nx, nz]
        t_grid = t.view(B, 1, 1, 1).expand(B, 1, self.nx, self.nz)

        # [B, 4 + 3 + 1, nx, nz] = [B, 8, nx, nz]
        model_input = torch.cat([u, grid, t_grid], dim=1)

        N_theta = self.fno(model_input)

        L_t = 50.0
        sigma_t = 1.0 - torch.exp(-t / L_t)
        sigma_t = sigma_t.view(B, 1, 1, 1)

        f_theta = self.g[None] + sigma_t * (
            self.H[None]
            - self.g[None]
            + self.D_x[None] * (N_theta - self.H[None])
        )

        return f_theta
    

