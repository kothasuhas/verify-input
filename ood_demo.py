import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

p = 0.9
H = torch.Tensor([[-1, -1, 1], [-1, -1, 1]])
thresh = np.log(p / (1 - p))
d = torch.Tensor([thresh, thresh])

model, H, d = load_model("toy", "test-weights.pt", H, d)

num_cs=20
input_lbs = [-2.0, -2.0]
input_ubs = [2.0, 2.0]

num_iters = 5

optimize(model, H, d, num_cs, input_lbs, input_ubs, num_iters, max_branching_depth=None, contour=False)
