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

orig_model = load_model("toy", "ood/models/test-weights.pt")

policy = toy_maxy()

# chosen such that y2 and y3 can be passed through the added ReLU without being changed
# can be computed e.g. using RSIP
M = 50.0

with torch.no_grad():
    M1 = torch.Tensor([[1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
    M2 = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    policy[1].weight = orig_model[1].weight
    policy[1].bias   = orig_model[1].bias
    policy[3].weight = orig_model[3].weight
    policy[3].bias   = orig_model[3].bias
    policy[5].weight = torch.nn.Parameter(M1.T @ orig_model[5].weight)
    policy[5].bias   = torch.nn.Parameter(M1.T @ orig_model[5].bias + torch.Tensor([0.0, M, M]))
    policy[7].weight = torch.nn.Parameter(M2.T)
    policy[7].bias   = torch.nn.Parameter(-torch.Tensor([M, M]))

torch.save(policy.state_dict(), 'ood/models/test-weights-maxy.pt')

print("Ensure the model correctly computes max(y1, y2), y3")
for _ in range(200):
    a = np.random.random() * 2 - 1
    b = np.random.random() * 2 - 1
    a = -5
    b = 1
    y1, y2, y3 = orig_model(torch.Tensor([[a, b]])).squeeze().tolist()
    z1, z2 = policy(torch.Tensor([[a, b]])).squeeze().tolist()
    assert np.allclose([max(y1, y2)], [z1], atol=0.0001), (y1, y2, z1)
    assert np.allclose([y3], [z2], atol=0.0001), (y3, z2)
print("Done")
