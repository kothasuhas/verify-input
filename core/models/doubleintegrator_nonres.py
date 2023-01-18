import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def doubleintegrator_nonres():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2, 12),
        nn.ReLU(),
        nn.Linear(12, 7),
        nn.ReLU(),
        nn.Linear(7, 2)
    )
    return model

def doubleintegrator_orig():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    return model

def doubleintegrator_nonres_ulimits():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2, 12),
        nn.ReLU(),
        nn.Linear(12, 7),
        nn.ReLU(),
        nn.Linear(7, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    )
    return model