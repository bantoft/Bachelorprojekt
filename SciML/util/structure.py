import torch

NN_STRUCTURE = {
    "input_size": 75,
    "output_size": 4,
    "output_names": ("lnn", "lnpe", "lnpi", "phi"),
    "layers": [
        {"size": 400, "non_lin_foo": torch.nn.Tanh},
        {"size": 200, "non_lin_foo": torch.nn.SiLU},
        {"size": 100, "non_lin_foo": torch.nn.Tanh},
        {"size": 100, "non_lin_foo": torch.nn.SiLU},
        {"size": 50, "non_lin_foo": torch.nn.Tanh},
    ],
}
