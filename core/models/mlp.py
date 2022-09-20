import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mlp():
    model = nn.Sequential(
        Flatten(),
        nn.utils.spectral_norm(nn.Linear(784, 20)),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model