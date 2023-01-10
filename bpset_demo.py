import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

H = torch.Tensor([
        [1, 0], 
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
d = torch.Tensor([
        -5.0,
        4.5,
        -0.25,
        -0.25
    ])

model, H, d = load_model("doubleintegrator_nonres", "doubleintegrator.pt", H, d, stack_n_times=3)

num_cs = 20
input_lbs = [-5.0, -5.0]
input_ubs = [5.0, 5.0]

num_iters = 5

optimize(model, H, d, num_cs, input_lbs, input_ubs, num_iters, perform_branching=True, contour=False)
