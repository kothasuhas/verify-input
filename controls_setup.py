# code to convert a keras controls policy into a torch nonresidual network
# (no need to run unless you have a new controls policy to test)

import torch
from torch import nn

from keras.models import model_from_json

from core.models.doubleintegrator_nonres import doubleintegrator_nonres, doubleintegrator_nonres_ulimits

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

u_limits = torch.Tensor([-1.0, 1.0])

with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
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

orig_model = nn.Sequential(
    Flatten(),
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
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
    # here, we assume no u_limits are used. If they are, this layer will be replaced
    policy[5].weight = torch.nn.Parameter(B.matmul(keras_layer(4)))
    policy[5].bias   = torch.nn.Parameter(B.matmul(keras_layer(5)))

    orig_model[1].weight = keras_layer(0)
    orig_model[1].bias   = keras_layer(1)
    orig_model[3].weight = keras_layer(2)
    orig_model[3].bias   = keras_layer(3)
    orig_model[5].weight = keras_layer(4)
    orig_model[5].bias   = keras_layer(5)
torch.save(orig_model.state_dict(), 'doubleintegrator_orig.pt')

if u_limits is not None:
    network_output_features = 1
    policy = policy[:-1]  # the last layer already incorporated B
    layer = nn.Linear(policy[-2].out_features, network_output_features, True)
    layer.weight = keras_layer(4)
    layer.bias = keras_layer(5)
    policy.append(layer)  # now the model outputs u, not Bu

    # Last b -= u_min
    policy[-1].bias.data -= u_limits[0]
    # ReLU
    policy.append(nn.ReLU())
    # W = -I, b = u_max - u_min
    layer = nn.Linear(network_output_features, network_output_features, True)
    layer.weight.data = -torch.eye(network_output_features)
    layer.bias.data = torch.Tensor(u_limits[1] - u_limits[0])
    policy.append(layer)
    # ReLU
    policy.append(nn.ReLU())
    # W = -I, b = u_max
    layer = nn.Linear(network_output_features, network_output_features, True)
    assert network_output_features == 1
    layer.weight.data = torch.nn.Parameter(-B)  # u is clipped, so B can be used now
    layer.bias.data = torch.tensor([u_limits[1]]).matmul(B.T)
    policy.append(layer)

print(policy)

if u_limits is not None:
    policy_nores = doubleintegrator_nonres_ulimits()
else:
    policy_nores = doubleintegrator_nonres()

print(policy_nores)

with torch.no_grad():
    for layer in policy_nores:
        if not isinstance(layer, nn.Linear):
            continue
        else:
            layer.weight = torch.nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    policy_nores[1].weight[:-2] = policy[1].weight
    policy_nores[1].weight[-2:] = torch.nn.Parameter(torch.eye(2))
    policy_nores[1].bias[:-2] = policy[1].bias
    policy_nores[1].bias[-2:] = torch.nn.Parameter(torch.Tensor([10.0, 10.0]))

    policy_nores[3].weight[:-2,:-2] = policy[3].weight
    policy_nores[3].weight[-2:,-2:] = torch.nn.Parameter(torch.eye(2))
    policy_nores[3].bias[:-2] = policy[3].bias

    if u_limits is not None:
        print(policy_nores[5].weight.shape, policy[5].weight.shape)
        policy_nores[5].weight[:-2,:-2] = policy[5].weight
        policy_nores[5].weight[-2:,-2:] = torch.nn.Parameter(torch.eye(2))
        policy_nores[5].bias[:-2] = policy[5].bias

        policy_nores[7].weight[:-2,:-2] = policy[7].weight
        policy_nores[7].weight[-2:,-2:] = torch.nn.Parameter(torch.eye(2))
        policy_nores[7].bias[:-2] = policy[7].bias

        policy_nores[9].weight[:,:-2] = policy[9].weight
        policy_nores[9].weight[:,-2:] = torch.nn.Parameter(A)
        policy_nores[9].bias = torch.nn.Parameter(policy[9].bias - A.matmul(torch.Tensor([10.0, 10.0])))
    else:
        policy_nores[5].weight[:,:-2] = policy[5].weight
        policy_nores[5].weight[:,-2:] = torch.nn.Parameter(A)
        policy_nores[5].bias = torch.nn.Parameter(policy[5].bias - A.matmul(torch.Tensor([10.0, 10.0])))

def forward(x, policy):
    return nn.functional.linear(x, A, bias=None) + policy(x)

print("Ensure the model combining Ax+Bu is equivalent")
print(forward(torch.Tensor([[1.0, 1.0]]), policy))
print(policy_nores(torch.Tensor([[1.0, 1.0]])))

if u_limits is None:
    torch.save(policy_nores.state_dict(), 'doubleintegrator.pt')
else:
    torch.save(policy_nores.state_dict(), 'doubleintegrator_ulimits1.pt')
