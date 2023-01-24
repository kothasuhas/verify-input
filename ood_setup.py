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

orig_model = load_model("toy", "log/01-24-18:27:20--TEST-3L/weights-last.pt")

policy = toy_maxy()

with torch.no_grad():
    M1 = torch.Tensor([[1., -1., -1., 1., 0., 0.], [1., -1., 1., -1., 0., 0.], [0., 0., 0., 0., 1., -1.]])
    M2 = torch.Tensor([[0.5, 0.], [-0.5, 0.], [0.5, 0.], [0.5, 0.], [0., 1.], [0., -1.]])
    policy[1].weight = orig_model[1].weight
    policy[1].bias   = orig_model[1].bias
    policy[3].weight = orig_model[3].weight
    policy[3].bias   = orig_model[3].bias
    policy[5].weight = torch.nn.Parameter(M1.T @ orig_model[5].weight)
    policy[5].bias   = torch.nn.Parameter(M1.T @ orig_model[5].bias)
    policy[7].weight = torch.nn.Parameter(M2.T)
    policy[7].bias   = torch.nn.Parameter(torch.zeros((2,)))

torch.save(policy.state_dict(), 'test-weights-maxy.pt')

print("Ensure the model correctly computes max(y1, y2), y3")
for _ in range(200):
    a = np.random.random() * 2 - 1
    b = np.random.random() * 2 - 1
    y1, y2, y3 = orig_model(torch.Tensor([[a, b]])).squeeze().tolist()
    z1, z2 = policy(torch.Tensor([[a, b]])).squeeze().tolist()
    assert np.allclose([max(y1, y2)], [z1], atol=0.0001), (y1, y2, z1)
    assert np.allclose([y3], [z2])
print("Done")
