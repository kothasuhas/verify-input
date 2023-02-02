import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# defines the bounding box [4.5, 5.0] x [-0.25, 0.25]
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

ALLOW_TO_INIT_WITH_BOUNDS_OF_PREV_MODEL = True
DONT_OPTIMIZE_LOADED_LAYERS = True
MAXIMAL_T = 10

for stack_n_times in range(1, MAXIMAL_T+1):
    init_with_bounds_of_prev_model = ALLOW_TO_INIT_WITH_BOUNDS_OF_PREV_MODEL and stack_n_times > 1
    model = load_model("doubleintegrator_nonres", "doubleintegrator.pt", stack_n_times=stack_n_times)

    num_cs = 20
    # The driver will bound the input based on cs *both* from above and below,
    # so we'll get 2*num_cs many lines and we don't want to give symmetrical cs values
    cs = [[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)]
    cs = torch.Tensor(cs)

    input_lbs = [-5.0, -5.0]
    input_ubs = [5.0, 5.0]

    max_num_iters = 30
    convergence_threshold = 0.005
    max_branching_depth = 0
    plotting_level = PlottingLevel.NO_PLOTTING

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
        load_bounds_of_stacked=stack_n_times-1 if init_with_bounds_of_prev_model else None,
        save_bounds_as_stacked=stack_n_times,
        dont_optimize_loaded_layers=DONT_OPTIMIZE_LOADED_LAYERS,
    )
