import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel

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

model = load_model("doubleintegrator_nonres_ulimits", "doubleintegrator_ulimits1.pt", stack_n_times=5)

num_cs = 20
cs = [[np.cos(2*np.pi*t / num_cs), np.sin(2*np.pi*t / num_cs)] for t in range(num_cs)]

input_lbs = [-5.0, -5.0]
input_ubs = [5.0, 5.0]

max_num_iters = 30
convergence_threshold = 0.005
max_branching_depth = 0
plotting_level = PlottingLevel.ALL_STEPS

optimize(
    model,
    H,
    d,
    cs,
    input_lbs,
    input_ubs,
    max_num_iters,
    convergence_threshold=convergence_threshold,
    max_branching_depth=max_branching_depth,
    plotting_level=plotting_level,
)
