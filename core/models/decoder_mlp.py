import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def decoder_mlp():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(20, 400),
        nn.ReLU(),
        nn.Linear(400, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model