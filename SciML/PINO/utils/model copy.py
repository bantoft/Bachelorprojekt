import torch

from neuralop.models.base_model import BaseModel


class WrappedFNO(BaseModel):
    FIELDS = ("lnn", "lnpe", "lnpi", "vort")

    def __init__(self, fno, info, phys):
        super().__init__()

        self.info = info
        self.phys = phys
        self.fno = fno

        self.dtype = self.info.dtype
        self.nx = self.info.parameters.num_x
        self.nz = self.info.parameters.num_z
        self.device = self.info.device
        self.lx = self.info.parameters.Lx
        self.lz = self.info.parameters.Lz

        x_line = torch.linspace(
            0.0,
            self.lx,
            self.nx,
            device=self.device,
            dtype=self.dtype,
        )
        z_line = torch.linspace(
            0.0,
            self.lz,
            self.nz,
            device=self.device,
            dtype=self.dtype,
        )

        self.register_buffer("x_line", x_line)
        self.register_buffer("z_line", z_line)

        for field_name in self.FIELDS:
            coeffs = self.solve_hermite_coeffs(field_name)
            self.register_buffer(f"{field_name}_hermite_coeffs", coeffs)

        grid = self.spatial_grid()
        self.register_buffer("grid", grid)

        g, H, D_x = self.static_constrains()
        self.register_buffer("g", g)
        self.register_buffer("H", H)
        self.register_buffer("D_x", D_x)

        self.last_coord = None
        self.last_embedded_grid = None

    def _x_grid(self, batch_size):
        return self.x_line.view(1, 1, self.nx, 1).expand(batch_size, 1, self.nx, self.nz)

    def _z_grid(self, batch_size):
        return self.z_line.view(1, 1, 1, self.nz).expand(batch_size, 1, self.nx, self.nz)

    def _expand_time(self, t, batch_size):
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t.view(batch_size, 1, 1, 1).expand(batch_size, 1, self.nx, self.nz)

    def spatial_grid(self):
        x = self.x_line.view(self.nx, 1).expand(-1, self.nz)
        z = self.z_line.view(1, self.nz).expand(self.nx, -1)
        theta = 2.0 * torch.pi * z / self.lz
        return torch.stack([x, torch.sin(theta), torch.cos(theta)], dim=0)

    def embed_coord(self, t, batch_size, return_coord=False):
        """
        Enforces periodicity in z by mapping:
            (t, x, z) -> (t, x, sin(2*pi*z/Lz), cos(2*pi*z/Lz))
        """
        x = self._x_grid(batch_size)
        z = self._z_grid(batch_size)
        t_grid = self._expand_time(t, batch_size)

        coord = torch.cat([x, z, t_grid], dim=1).detach().clone().requires_grad_(True)

        x = coord[:, 0:1]
        z = coord[:, 1:2]
        t_grid = coord[:, 2:3]
        theta = 2.0 * torch.pi * z / self.lz
        grid = torch.cat([t_grid, x, torch.sin(theta), torch.cos(theta)], dim=1)

        self.last_coord = coord
        self.last_embedded_grid = grid

        if return_coord:
            return grid, coord

        return grid

    def solve_hermite_coeffs(self, field_name):
        bc = self.phys.bc[field_name]

        n = len(bc)
        powers = torch.arange(n, device=self.device, dtype=self.dtype)

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
                rhs = self.lx * value
            else:
                raise ValueError(f"Unknown bc_type: {bc_type}")

            M_rows.append(row)
            c_rows.append(rhs)

        M = torch.stack(M_rows)
        c = torch.stack(c_rows)
        return torch.linalg.solve(M, c)

    def hermit_x(self, field_name, x):
        """
        Compute Hermite polynomial h that satisfies boundary conditions in x.
        """
        coeffs = getattr(self, f"{field_name}_hermite_coeffs")
        powers = torch.arange(coeffs.shape[0], device=x.device, dtype=x.dtype).view(1, -1, 1, 1)

        alpha = self.x_line[0]
        beta = self.x_line[-1]
        s = (x - alpha) / (beta - alpha)
        basis = s.pow(powers)
        return (basis * coeffs.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

    def constraint_envelope(self, x):
        L_alpha = 0.5
        L_beta = 0.5

        sigma_alpha = 1.0 - torch.exp(-(x - self.x_line[0]) / L_alpha)
        sigma_beta = 1.0 - torch.exp(-(self.x_line[-1] - x) / L_beta)
        return sigma_alpha**2 * sigma_beta**2

    def static_constrains(self):
        g = torch.stack([
            self.phys.ic["init_lnn"],
            self.phys.ic["init_lnpe"],
            self.phys.ic["init_lnpi"],
            self.phys.ic["init_vort"],
        ], dim=0).to(device=self.device, dtype=self.dtype)

        g = g[:, :, None].expand(-1, -1, self.nz)
        x = self._x_grid(batch_size=1)
        H = torch.cat(
            [self.hermit_x(field_name, x=x) for field_name in self.FIELDS],
            dim=1,
        ).squeeze(0)
        D_x = self.constraint_envelope(x=x).squeeze(0)
        return g, H, D_x

    def make_constrains(self, x):
        H = torch.cat(
            [self.hermit_x(field_name, x=x) for field_name in self.FIELDS],
            dim=1,
        )
        D_x = self.constraint_envelope(x=x)
        return self.g, H, D_x

    def forward(self, u, t, return_coord=False):
        u = u.to(device=self.device, dtype=self.dtype)
        batch_size = u.shape[0]

        grid, coord = self.embed_coord(
            t=t,
            batch_size=batch_size,
            return_coord=True,
        )
        x = coord[:, 0:1]
        t_grid = coord[:, 2:3]

        model_input = torch.cat([u, grid], dim=1)
        N_theta = self.fno(model_input)

        L_t = 50.0
        sigma_t = 1.0 - torch.exp(-t_grid[:, :, :1, :1] / L_t)

        _, H, D_x = self.make_constrains(x=x)

        f_theta = self.g[None] + sigma_t * (
            H
            - self.g[None]
            + D_x * (N_theta - H)
        )

        if return_coord:
            return f_theta, coord

        return f_theta
