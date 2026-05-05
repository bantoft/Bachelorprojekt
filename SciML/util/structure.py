import torch

NN_STRUCTURE = {
    "input_size": 75,
    "output_size": 4,
    "output_names": ("lnn", "lnpe", "lnpi", "phi"),
    "layers": [
        {"size": 32, "non_lin_foo": torch.nn.Tanh},
        {"size": 16, "non_lin_foo": torch.nn.SiLU},
        {"size": 32, "non_lin_foo": torch.nn.SiLU},
    ],
}
