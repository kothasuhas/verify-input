# code to convert a keras controls policy into a torch nonresidual network
# (no need to run unless you have a new controls policy to test)

import torch
from torch import nn
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel
from core.models.toy import toy, toy_maxy

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

orig_model = load_model("toy", "test-weights.pt")

policy = toy_maxy()

with torch.no_grad():
    M1 = torch.Tensor([[1., 1., 0., 0., 0.], [-1., 0., -1., 0., 0.], [0., 0., 0., 1., -1.]])
    M2 = torch.Tensor([[1., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    policy[1].weight = orig_model[1].weight
    policy[1].bias   = orig_model[1].bias
    policy[3].weight = orig_model[3].weight
    policy[3].bias   = orig_model[3].bias
    policy[5].weight = torch.nn.Parameter(M1.T @ orig_model[5].weight)
    policy[5].bias   = torch.nn.Parameter(M1.T @ orig_model[5].bias)
    policy[7].weight = torch.nn.Parameter(M2)
    policy[7].bias   = torch.nn.Parameter(torch.zeros((2,)))
torch.save(orig_model.state_dict(), 'test-weights-maxy.pt')


print("Ensure the model correctly computes max(y1, y2), y3")
print(orig_model(torch.Tensor([[1.0, 1.0]])))
print(policy(torch.Tensor([[1.0, 1.0]])))
