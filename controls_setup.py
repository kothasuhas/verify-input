# code to convert a keras controls policy into a torch nonresidual network
# (no need to run unless you have a new controls policy to test)

import torch
from torch import nn

from keras.models import model_from_json
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

policy = nn.Sequential(
    Flatten(),
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

def keras_layer(i):
    if i % 2 == 0:
        return torch.nn.Parameter(torch.Tensor(loaded_model.weights[i].numpy()).transpose(0, 1))
    else:
        return torch.nn.Parameter(torch.Tensor(loaded_model.weights[i].numpy()))

A = torch.Tensor([[1.0, 1.0], [0.0, 1.0]])
B = torch.Tensor([[0.5], [1.0]])

with torch.no_grad():
    policy[1].weight = keras_layer(0)
    policy[1].bias   = keras_layer(1)
    policy[3].weight = keras_layer(2)
    policy[3].bias   = keras_layer(3)
    policy[5].weight = torch.nn.Parameter(B.matmul(keras_layer(4)))
    policy[5].bias   = torch.nn.Parameter(B.matmul(keras_layer(5)))

policy_nores = nn.Sequential(
    Flatten(),
    nn.Linear(2, 12),
    nn.ReLU(),
    nn.Linear(12, 7),
    nn.ReLU(),
    nn.Linear(7, 2)
)

with torch.no_grad():
    policy_nores[1].weight = torch.nn.Parameter(torch.zeros(12, 2))
    policy_nores[1].bias   = torch.nn.Parameter(torch.zeros(12))
    policy_nores[3].weight = torch.nn.Parameter(torch.zeros(7, 12))
    policy_nores[3].bias   = torch.nn.Parameter(torch.zeros(7))
    policy_nores[5].weight = torch.nn.Parameter(torch.zeros(2, 7))
    policy_nores[5].bias   = torch.nn.Parameter(torch.zeros(2))

    policy_nores[1].weight[:-2] = policy[1].weight
    policy_nores[1].weight[-2:] = torch.nn.Parameter(torch.eye(2))
    policy_nores[1].bias[:-2] = policy[1].bias
    policy_nores[1].bias[-2:] = torch.nn.Parameter(torch.Tensor([10.0, 10.0]))

    policy_nores[3].weight[:-2,:-2] = policy[3].weight
    policy_nores[3].weight[-2:,-2:] = torch.nn.Parameter(torch.eye(2))
    policy_nores[3].bias[:-2] = policy[3].bias

    policy_nores[5].weight[:,:-2] = policy[5].weight
    policy_nores[5].weight[:,-2:] = torch.nn.Parameter(A)
    policy_nores[5].bias = torch.nn.Parameter(policy[5].bias - A.matmul(torch.Tensor([10.0, 10.0])))


def forward(x, policy):
    return nn.functional.linear(x, A, bias=None) + policy(x)

print(forward(torch.Tensor([[1.0, 1.0]]), policy))
print(policy_nores(torch.Tensor([[1.0, 1.0]])))

torch.save(policy_nores.state_dict(), 'doubleintegrator.pt')
