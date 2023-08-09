import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

bounding_box = [[-1,1],[-1,1],[1.5,3.5],[-1,1],[-1,1],[-1,1]]
# SET GAMMA LEARNING_RATE TO BE 0.0001

# defines the above bounding box
H = torch.Tensor([
        [-1 if i == coord else 0 for i in range(len(bounding_box))]
        for coord in range(len(bounding_box))
    ] + [
        [1 if i == coord else 0 for i in range(len(bounding_box))]
        for coord in range(len(bounding_box))
    ])
d = torch.Tensor([
        bound[0] for bound in bounding_box
    ] + [
        -bound[1] for bound in bounding_box
    ])

print(H, d)


ALLOW_TO_INIT_WITH_BOUNDS_OF_PREV_MODEL = True
DONT_OPTIMIZE_LOADED_LAYERS = True
MAXIMAL_T = 1

for stack_n_times in range(1, MAXIMAL_T+1):
    init_with_bounds_of_prev_model = ALLOW_TO_INIT_WITH_BOUNDS_OF_PREV_MODEL and stack_n_times > 1
    model = load_model("quadrotor_nonres_ulimits", "control/models/quadrotor_ulimits.pt", stack_n_times=stack_n_times)

    num_cs = 20
    # The driver will bound the input based on cs *both* from above and below,
    # so we'll get 2*num_cs many lines and we don't want to give symmetrical cs values
    cs = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ]
    cs = torch.Tensor(cs)

    # bounds derived from controls, same bounds utilized by quadrotor example
    # in https://github.com/neu-autonomy/nfl_veripy/blob/release/src/nfl_veripy/_static/example_configs/ojcsys23/quadrotor_hybreach.yaml
    input_lbs = [lim[0] - 1 for lim in bounding_box[:3]] + [-1.0, -1.0, -1.0]
    input_ubs = [lim[1] + 1 for lim in bounding_box[:3]] + [1.0, 1.0, 1.0]

    print(input_lbs, input_ubs)

    max_num_iters = 500
    convergence_threshold = 100.0
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
