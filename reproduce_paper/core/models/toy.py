import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def toy():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 3)
    )
    return model

def toy_maxy():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 3),
        nn.ReLU(),
        nn.Linear(3, 2)
    )
    return model
