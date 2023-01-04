import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model

model = load_model("doubleintegrator_nonres", "doubleintegrator.pt")

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

cs = [np.random.normal(size=2) for _ in range(20)]
input_lbs = [-5.0, -5.0]
input_ubs = [5.0, 5.0]

num_iters = 5

optimize(model, H, d, cs, input_lbs, input_ubs, num_iters, perform_branching=False, contour=False)
