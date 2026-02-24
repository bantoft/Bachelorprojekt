import torch

from PDE_system import c

def pde_residuals(model, x_f, t_f, c=c):
    # Sørg for at autograd kan differentiere w.r.t. x_f, t_f
    x_f = x_f.clone().detach().requires_grad_(True)
    t_f = t_f.clone().detach().requires_grad_(True)

    u_pred, v_pred = model(x_f, t_f)

    # Førsteordens afledte af u
    grads_u = torch.autograd.grad(
        u_pred, [x_f, t_f],
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True
    )
    u_x = grads_u[0]
    u_t = grads_u[1]

    # Anden afledt mht. x: u_xx
    u_xx = torch.autograd.grad(
        u_x, x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # Førsteordens afledt af v
    grads_v = torch.autograd.grad(
        v_pred, [x_f, t_f],
        grad_outputs=torch.ones_like(v_pred),
        create_graph=True
    )
    v_x = grads_v[0]
    v_t = grads_v[1]

    # PDE residualer
    r1 = u_t - v_pred
    r2 = v_t - (c**2) * u_xx

    return r1, r2