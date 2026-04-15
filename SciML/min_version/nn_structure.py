import torch.nn as nn

nn_mapping = {
    'activation_functions' : [nn.ELU(), nn.Gelu(), nn.LogSigmoid(), nn.Sigmoid(), nn.SiLU(),nn.Tanh(), nn.Tanhshrink()]
    'layer_sizes': [16, 32, 64, 128, 256, 512],
}

# Definer structuren med mapping til nn_mapping
NN_structure = {
    'input':3,
    'output':4,
    'layer1': {'size': 1, 'activation': 1},
    'layer2': {'size': 1, 'activation': 1},
    'layer3': {'size': 1, 'activation': 1},
    'layer4': {'size': 1, 'activation': 1},
}