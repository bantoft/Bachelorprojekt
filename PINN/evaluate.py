import torch
from PDE_system import device, exact_u, exact_v



def evaluate_model(model, N_x=100, N_t=100):
    x = torch.linspace(0, 1, N_x).view(-1, 1)
    t = torch.linspace(0, 1, N_t).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_grid = X.reshape(-1, 1).to(device)
    t_grid = T.reshape(-1, 1).to(device)

    with torch.no_grad():
        u_pred, v_pred = model(x_grid, t_grid)
        u_exact_vals = exact_u(x_grid, t_grid)
        v_exact_vals = exact_v(x_grid, t_grid)

    # L2 fejl
    u_error = torch.sqrt(torch.mean((u_pred - u_exact_vals)**2)).item()
    v_error = torch.sqrt(torch.mean((v_pred - v_exact_vals)**2)).item()
    print(f"L2 fejl: u={u_error:.4e}, v={v_error:.4e}")

    return (X.cpu().numpy(), T.cpu().numpy(),
            u_pred.cpu().numpy().reshape(N_x, N_t),
            u_exact_vals.cpu().numpy().reshape(N_x, N_t))