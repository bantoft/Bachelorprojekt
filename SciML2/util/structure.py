import torch

NN_STRUCTURE = {
    "input_size": 3, # Giv alle billeder: 4 kanaler: [128x128x4] -> 3 input features (x, z)
    "output_size": 4, # [128x128x4] -> 3 input features (x, z)
    "output_names": ("lnn", "lnpe", "lnpi", "phi"),
    "layers": [
        {"size": 64, "non_lin_foo": torch.nn.Tanh},
        {"size": 64, "non_lin_foo": torch.nn.SiLU},
    ],
}