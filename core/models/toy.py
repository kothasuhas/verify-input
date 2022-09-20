import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# def toy():
#     model = nn.Sequential(
#         Flatten(),
#         nn.Linear(2, 40),
#         nn.ReLU(),
#         nn.Linear(40, 40),
#         nn.ReLU(),
#         nn.Linear(40, 2)
#     )
#     return model

def toy():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 2)
    )
    return model
