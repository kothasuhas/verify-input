import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unsqueeze(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            return x.unsqueeze(1)
        else:
            return x

# def convnet():
#     model = nn.Sequential(
#         nn.Conv2d(3, 32, 3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 32, 4, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, 3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(64, 64, 4, stride=2, padding=1),
#         nn.ReLU(),
#         Flatten(),
#         nn.Linear(64*8*8, 512),
#         nn.ReLU(),
#         nn.Linear(512, 512),
#         nn.ReLU(),
#         nn.Linear(512, 10)
#     )
#     return model

def convnet():
    width = 128

    model = nn.Sequential(
        Unsqueeze(),
        nn.Conv2d(1, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width * 2, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width * 2, width * 2, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width*2*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model