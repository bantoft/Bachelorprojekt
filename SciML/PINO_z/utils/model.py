import torch

from neuralop.models.base_model import BaseModel


class WrappedFNO(BaseModel):

    def __init__(self, fno, info, phys, m: int = 3):
        super().__init__()

        if m <= 0 or m % 2 != 1:
            raise ValueError("m skal vaere et positivt ulige tal.")

        self.m = m
        self.info = info
        self.phys = phys
        self.fno = fno

        self.dtype = self.info.dtype
        self.device = self.info.device

        self.nx = self.info.parameters.num_x
        self.nz = self.info.parameters.num_z
        self.nt = self.info.parameters.num_t # Antal produceret t

        self.lx = self.info.parameters.Lx
        self.lz = self.info.parameters.Lz
        self.lt = self.info.settings["root", "t_end"] # totalt længde af fysiske tidsdomæne

        # cord buffer til at holde fysiske koordinater x, z og t
        coord = torch.empty(self.nx, self.m, 3, device=self.device, dtype=self.dtype)
        # x koordinat
        coord[..., 0] = torch.arange(self.nx, device=self.device, dtype=self.dtype).view(self.nx, 1) # x koordinat
        # z koordinat
        coord[..., 1] = torch.arange(self.m, device=self.device, dtype=self.dtype).view(1, self.m) # z koordinat
        # t skalar
        coord[..., 2] = 0.0
        self.register_buffer("coord_base", coord)



        self.time_scale = 50.0

        g, H, D_x = self.make_x_constrains()
        self.register_buffer("g", g)
        self.register_buffer("H", H)
        self.register_buffer("D_x", D_x)


    ## Lav ny version der embedder x,z og t så det stemmer mere overens med PINO_full
    def embed_cord(self, u_subset, t, z):
        """
        Embedder de fysiske koordinater x, z og t
        x (Linear standardisering): [0, Lx] -> [-1, 1]:     x' = 2 * x / Lx - 1
        z (Fourier embedding):      [0, Lz] -> [z_1, z_2]:  z' = [cos(2 * pi * z / Lz), sin(2 * pi * z / Lz)] 
        t (Linear standardisering): [0, Lt] -> [0, 1]:      t' = t / Lt
        """
        batch_size, _, nx, m = u_subset.shape # [Bach, fields = 4, nx, m]
        coord_fysisk = self.coord_base.detach().clone() # [nx, m, 3]

        coord_fysisk[..., 1] = z.view(1, m)
        coord_fysisk[..., 2] = t

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
        D_x = D_x[None, :, None].expand(1, self.nx, self.m)

        g = torch.stack([
            self.phys.ic["init_lnn"],
            self.phys.ic["init_lnpe"],
            self.phys.ic["init_lnpi"],
            self.phys.ic["init_phi"],
        ], dim=0).to(dtype=self.dtype)
        g = g[:, :, None].expand(-1, -1, self.m)

        H = torch.stack([
            self.hermit_x("lnn"),
            self.hermit_x("lnpe"),
            self.hermit_x("lnpi"),
            self.hermit_x("phi"),
        ], dim=0).to(dtype=self.dtype)
        H = H[:, :, None].expand(-1, -1, self.m)

        return g, H, D_x






        return x_hat, z_hat, t_hat

    # def create_input(self, u_subset, t):
    #     batch_size, _, _, _ = u_subset.shape

    #     x_coord = self.x_line.view(1, 1, self.nx, 1).expand(batch_size, 1, self.nx, self.m)
    #     z_coord = self.z_local.expand(batch_size, 1, self.nx, self.m)
    #     t_coord = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, self.nx, self.m)

    #     coord = torch.cat([x_coord, z_coord, t_coord], dim=1).requires_grad_(True)

    #     x_coord = coord[:, 0:1]
    #     z_coord = coord[:, 1:2]
    #     t_coord = coord[:, 2:3]
        
    #     z1 = torch.sin(2.0 * torch.pi * z_coord / self.lz_tensor)
    #     z2 = torch.cos(2.0 * torch.pi * z_coord / self.lz_tensor)

    #     model_input = torch.cat([u_subset, x_coord, z1, z2, t_coord], dim=1)
    #     return model_input, coord

    # def forward(self, u_subset, t):
    #     model_input, coord = self.create_input(u_subset, t)

    #     n_theta = self.fno(model_input)

    #     sigma_t = (1.0 - torch.exp(-t / self.time_scale)).view(t.shape[0], 1, 1, 1)
    #     f = self.g[None] + sigma_t * (self.H[None] - self.g[None] + self.D_x[None] * (n_theta - self.H[None]))

    #     f_theta = {"lnn": f[:, 0:1], "lnpe": f[:, 1:2], "lnpi": f[:, 2:3], "phi": f[:, 3:4]}

    #     return f_theta, coord
