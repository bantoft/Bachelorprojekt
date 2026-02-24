import torch
import torch.nn as nn

from residual import pde_residuals

mse_loss = nn.MSELoss()

def pinn_loss(model, 
              x_f, t_f, 
              x_0, t_0, u_0, v_0,
              x_b1, t_b1, u_b1, v_b1,
              x_b2, t_b2, u_b2, v_b2,
              w_f=1.0, w_0=1.0, w_b=1.0):

    # PDE residual på kollokationspunkter
    r1, r2 = pde_residuals(model, x_f, t_f)
    loss_pde = mse_loss(r1, torch.zeros_like(r1)) + \
               mse_loss(r2, torch.zeros_like(r2))

    # Initialbetingelser
    u0_pred, v0_pred = model(x_0, t_0)
    loss_ic = mse_loss(u0_pred, u_0) + mse_loss(v0_pred, v_0)

    # Randbetingelser venstre
    u_b1_pred, v_b1_pred = model(x_b1, t_b1)
    loss_bc1 = mse_loss(u_b1_pred, u_b1) + mse_loss(v_b1_pred, v_b1)

    # Randbetingelser højre
    u_b2_pred, v_b2_pred = model(x_b2, t_b2)
    loss_bc2 = mse_loss(u_b2_pred, u_b2) + mse_loss(v_b2_pred, v_b2)

    loss_bc = loss_bc1 + loss_bc2

    total_loss = w_f * loss_pde + w_0 * loss_ic + w_b * loss_bc
    return total_loss, loss_pde, loss_ic, loss_bc