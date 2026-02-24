import torch
import numpy as np

# Enhedsenhed (CPU); brug "cuda" hvis du har GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bølge-hastighed
c = 1.0
pi = np.pi

def exact_u(x, t, c=c):
    # x, t: torch tensors med samme shape
    return torch.sin(pi * x) * torch.cos(c * pi * t)

def exact_v(x, t, c=c):
    return -c * pi * torch.sin(pi * x) * torch.sin(c * pi * t)

# Sample kollokationspunkter (interior)
def sample_collocation_points(N_f):
    x_f = torch.rand(N_f, 1)  # Uniform i [0,1]
    t_f = torch.rand(N_f, 1)  # Uniform i [0,1]
    return x_f.to(device), t_f.to(device)

# Sample initialbetingelser
def sample_initial_points(N_0):
    x_0 = torch.rand(N_0, 1)
    t_0 = torch.zeros_like(x_0)
    u_0 = exact_u(x_0, t_0)  # sin(pi x)
    v_0 = exact_v(x_0, t_0)  # = 0
    return x_0.to(device), t_0.to(device), u_0.to(device), v_0.to(device)

# Sample randbetingelser
def sample_boundary_points(N_b):
    # venstre rand x=0
    t_b1 = torch.rand(N_b, 1)
    x_b1 = torch.zeros_like(t_b1)
    # højre rand x=1
    t_b2 = torch.rand(N_b, 1)
    x_b2 = torch.ones_like(t_b2)

    u_b1 = exact_u(x_b1, t_b1)
    v_b1 = exact_v(x_b1, t_b1)
    u_b2 = exact_u(x_b2, t_b2)
    v_b2 = exact_v(x_b2, t_b2)

    return (x_b1.to(device), t_b1.to(device), u_b1.to(device), v_b1.to(device),
            x_b2.to(device), t_b2.to(device), u_b2.to(device), v_b2.to(device))
