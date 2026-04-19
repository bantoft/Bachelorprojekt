import torch.nn as nn

from Min_SciML.model import PINN

nn_mapping = {
    'activation_functions' : [nn.ELU(), nn.GELU(), nn.LogSigmoid(), nn.Sigmoid(), nn.SiLU(),nn.Tanh(), nn.Tanhshrink()],
    'layer_sizes': [16, 32, 64, 128, 256, 512],
}

# Definer structuren med mapping til nn_mapping
NN_structure = {
    'input_size':3,
    'output_size':4,
    'layer1': {'size': 1, 'activation': 1},
    'layer2': {'size': 1, 'activation': 1},
    'layer3': {'size': 1, 'activation': 1},
    'layer4': {'size': 1, 'activation': 1},
}


import torch
import torch.nn as nn


# class PINN(nn.Module):
#     def __init__(self, nn_structure: dict):
#         super().__init__()

#         input_size = int(nn_structure.get("input_size", 3))
#         output_size = int(nn_structure.get("output_size", 1))

def test(nn_structure):
    layers = []
    prev_size = int(nn_structure.get("input_size", 3))
    for i in nn_structure:
        if i == "input_size" or i == "output_size": continue
        print(i)

test(NN_structure)