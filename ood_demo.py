import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

p = 0.9
H = torch.Tensor([[-1, -1, 1], [-1, -1, 1]])
thresh = np.log(p / (1 - p))
d = torch.Tensor([thresh, thresh])

model = load_model("toy", "test-weights.pt")

num_cs=20
# The driver will bound the input based on cs *both* from above and below,
# so we don't want to give symmetrical cs values
# Also, we'll get 2*num_cs many lines in our output
cs = [[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)]
cs = torch.Tensor(cs)

input_lbs = [-2.0, -2.0]
input_ubs = [2.0, 2.0]

max_num_iters = 10
convergence_threshold = 0.05
max_branching_depth = 1
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
