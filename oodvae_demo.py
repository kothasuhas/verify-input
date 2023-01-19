import torch
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel
from crown.lp import get_optimized_grb_result, get_triangle_grb_model, get_optimal_grb_model
from tqdm import tqdm

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# assert that -class1 + classk + 20.0 <= 0
H = torch.zeros(9, 10)
H[:,0] = -1.0
H[:,1:] = torch.eye(9)

d = torch.full((9,), 20.0)

model = load_model("decoder_mlp", "decoder_mlp.pt")

num_cs = 20
cs = [torch.randn(20) for _ in range(num_cs)]

input_lbs = [-5.0 for _ in range(20)]
input_ubs = [5.0 for _ in range(20)]

max_num_iters = 10
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
)
