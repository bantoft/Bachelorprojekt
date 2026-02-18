from setup_utils import Dynamics

class SimpleSystem2D:
    def __init__(self, grid, Da=0.1, Db=0.1, alpha=1.0, vx=0.0, vy=0.0):
        self.grid = grid
        self.Da = Da
        self.Db = Db
        self.alpha = alpha
        self.vx = vx
        self.vy = vy

        # preallocate work arrays (undgår allocations i time loop)
        self.ax = grid.alloc_phys()
        self.ay = grid.alloc_phys()
        self.bx = grid.alloc_phys()
        self.by = grid.alloc_phys()

        self.lapa = grid.alloc_phys()
        self.lapb = grid.alloc_phys()

    def rhs(self, a, b):
        """
        a, b er FELTER med ghost cells
        returnerer fysiske RHS arrays (uden ghosts)
        """

        g = self.grid

        # 1️⃣ Opdater randbetingelser
        Dynamics.apply_periodic_bc(a)
        Dynamics.apply_periodic_bc(b)

        # 2️⃣ Første afledte
        Dynamics.dx_central(a, g.dx, self.ax)
        Dynamics.dy_central(a, g.dy, self.ay)

        Dynamics.dx_central(b, g.dx, self.bx)
        Dynamics.dy_central(b, g.dy, self.by)

        # 3️⃣ Laplace = d2x + d2y
        tmp = g.alloc_phys()  # kun for clarity (kan optimeres væk)
        Dynamics.d2x_central(a, g.dx, self.lapa)
        Dynamics.d2y_central(a, g.dy, tmp)
        self.lapa += tmp

        Dynamics.d2x_central(b, g.dx, self.lapb)
        Dynamics.d2y_central(b, g.dy, tmp)
        self.lapb += tmp

        # 4️⃣ RHS konstruktion
        rhs_a = (
            - self.vx * self.ax
            - self.vy * self.ay
            + self.Da * self.lapa
            + self.alpha * g.interior(b)
        )

        rhs_b = (
            - self.vx * self.bx
            - self.vy * self.by
            + self.Db * self.lapb
            - self.alpha * g.interior(a)
        )
        # TODO
            # returner liste med rhs også implimenter dt her
        return rhs_a, rhs_b
