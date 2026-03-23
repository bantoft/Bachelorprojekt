import torch

from PDE_system import (
    sample_boundary_points,
    sample_collocation_points,
    sample_initial_points,
)
from PINN import model
from loss import pinn_loss

# Antal punkter
N_f = 10_000  # kollokationspunkter
N_0 = 200     # initialbetingelser
N_b = 200     # randbetingelser pr. side

# Sample data
x_f, t_f = sample_collocation_points(N_f)
x_0, t_0, u_0, v_0 = sample_initial_points(N_0)
x_b1, t_b1, u_b1, v_b1, x_b2, t_b2, u_b2, v_b2 = sample_boundary_points(N_b)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_iterations = 5000  # juster efter behov

for it in range(num_iterations):
    optimizer.zero_grad()

    total_loss, loss_pde, loss_ic, loss_bc = pinn_loss(
        model,
        x_f, t_f,
        x_0, t_0, u_0, v_0,
        x_b1, t_b1, u_b1, v_b1,
        x_b2, t_b2, u_b2, v_b2,
        w_f=1.0, w_0=1.0, w_b=1.0
    )

    total_loss.backward()
    optimizer.step()

    if it % 500 == 0:
        print(f"Iter {it}: total={total_loss.item():.4e}, "
              f"PDE={loss_pde.item():.4e}, "
              f"IC={loss_ic.item():.4e}, "
              f"BC={loss_bc.item():.4e}")