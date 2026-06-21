import torch

NN_STRUCTURE = {
    "input_size": 75,
    "output_size": 4,
    "output_names": ("lnn", "lnpe", "lnpi", "phi"),
    "layers": [
        {"size": 2**8, "non_lin_foo": torch.nn.SiLU},
        {"size": 2**7, "non_lin_foo": torch.nn.SiLU},
        {"size": 2**7, "non_lin_foo": torch.nn.SiLU},
        {"size": 2**7, "non_lin_foo": torch.nn.Tanh},
        {"size": 2**6, "non_lin_foo": torch.nn.SiLU},
    ],
}
