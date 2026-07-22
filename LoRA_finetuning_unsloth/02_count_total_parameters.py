def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    count = 0
    for param in model.parameters():
        count += param.numel() #  numel: counts how many numbers are inside the tensor.
    return count


import torch.nn as nn
m = nn.Linear(4, 3)  # 4*3 weights + 3 bias = 15
count_total_parameters(m)