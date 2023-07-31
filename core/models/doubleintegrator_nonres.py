import torch.nn as nn

STATE_DIM = 2
HIDDEN1_DIM = 10
HIDDEN2_DIM = 5
POLICY_DIM = 1

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def doubleintegrator_orig():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM, HIDDEN1_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM, POLICY_DIM)
    )
    return model

def doubleintegrator_nonres():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM, HIDDEN1_DIM + STATE_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM + STATE_DIM, HIDDEN2_DIM + STATE_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM + STATE_DIM, STATE_DIM)
    )
    return model

def doubleintegrator_nonres_ulimits():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM, HIDDEN1_DIM + STATE_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM + STATE_DIM, HIDDEN2_DIM + STATE_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM + STATE_DIM, POLICY_DIM + STATE_DIM),
        nn.ReLU(),
        nn.Linear(POLICY_DIM + STATE_DIM, POLICY_DIM + STATE_DIM),
        nn.ReLU(),
        nn.Linear(POLICY_DIM + STATE_DIM, STATE_DIM),
    )
    return model