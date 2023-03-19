import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

models = [load_model(f"ffn{i}", f"clean/ffn{i}.pt") for i in range(2, 9)]
model = models[-2]

for property_num in range(50):
    print(f' PROPERTY {property_num}')
    with open(f'mnist_properties/mnist_{property_num}_local_property.in') as f:
        property_file = f.readlines()
        input_center = torch.Tensor(list(map(float, property_file[:784])))
        property_matrix = property_file[784:]
        property_matrix = list(map(lambda x : list(map(float, x.split(' '))), property_matrix))
        radius = 0.037

    num_cs=20
    # The driver will bound the input based on cs *both* from above and below,
    # so we don't want to give symmetrical cs values
    # Also, we'll get 2*num_cs many lines in our output
    cs = [[0.0 for _ in range(784)] for _ in range(1)]
    cs = torch.Tensor(cs)

    for attack_class in range(9):
        H = torch.Tensor(property_matrix)[attack_class:attack_class+1, :-1] * -1

        d = torch.Tensor(property_matrix)[attack_class:attack_class+1, -1]
        assert torch.all(d == 0)
        input_lbs = torch.clamp(input_center - radius, min=0.)
        input_ubs = torch.clamp(input_center + radius, max=1.)

        max_num_iters = 5
        convergence_threshold = 0.0
        max_branching_depth = 0
        plotting_level = PlottingLevel.NO_PLOTTING

        input_not_empty = optimize(
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
            return_success=True,
        )
        if input_not_empty:
            break
    if input_not_empty:
        print("Not verified")
    else:
        print("Verified")
