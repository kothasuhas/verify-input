import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def featurizer():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(3072, 3072),
        nn.LeakyReLU(negative_slope=0.5),
        nn.Linear(3072, 3072),
        nn.LeakyReLU(negative_slope=0.5),
        nn.Linear(3072, 3072),
        nn.LeakyReLU(negative_slope=0.5),
        nn.Linear(3072, 3072)
    )
    return model