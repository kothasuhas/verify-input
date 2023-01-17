import torch
from crown.driver import optimize
from crown.model_utils import load_model

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

num_iters = 10

optimize(model, H, d, cs, input_lbs, input_ubs, num_iters, perform_branching=False, plot=False)
