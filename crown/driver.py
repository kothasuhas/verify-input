from typing import List

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model_utils import get_num_layers, get_num_neurons, get_direction_layer_pairs, load_model
from .lp import get_optimized_grb_result, get_triangle_grb_model
from .crown import initialize_all, optimize_bound, ApproximatedInputBound, InputBranch
from .plot_utils import plot2d

def optimize(model, H, d, cs, input_lbs, input_ubs, num_iters, perform_branching=True, contour=True):
    plt.ion()
    plt.show()

    # Output the Gurobi-Text now
    gp.Model()

    approximated_input_bounds: List[ApproximatedInputBound] = []

    def get_initial_input_branch(model, H, d):
        resulting_lbs, resulting_ubs, params_dict, weights, biases = initialize_all(model=model, input_lbs=torch.Tensor(input_lbs), input_ubs=torch.Tensor(input_ubs), H=H, d=d)
        initial_input_branch = InputBranch(input_lbs=input_lbs, input_ubs=input_ubs, params_dict=params_dict, resulting_lbs=resulting_lbs, resulting_ubs=resulting_ubs, weights=weights, biases=biases)
        return initial_input_branch

    branches = [get_initial_input_branch(model, H, d)]
    if perform_branching:
        branches += branches[0].split()

    plot_number = 0
    num_layers = get_num_layers(model)
    assert num_layers == 3, "Proceed with caution, models with more/less than 3 layers were not tested so far"
    for branch in tqdm(branches, desc="Input Branches"):
        pbar = tqdm(range(num_iters), leave=False)
        last_b = []
        abort = False
        pending_approximated_input_bounds: List[ApproximatedInputBound] = []
        for _ in pbar:
            if abort:
                break
            pending_approximated_input_bounds = []
            pbar.set_description(f"Best solution to first bound: {last_b}")
            for direction, layeri in tqdm(get_direction_layer_pairs(model), desc="Directions & Layers", leave=False):
                if abort:
                    break
                neurons = get_num_neurons(model, layeri)
                for neuron in tqdm(range(neurons), desc="Neurons", leave=False):
                    if abort:
                        break
                    gamma = branch.params_dict[direction][layeri][neuron]['gamma']
                    alphas = branch.params_dict[direction][layeri][neuron]['alphas']
                    optim = torch.optim.SGD([
                        {'params': gamma, 'lr' : 0.0001}, 
                        {'params': alphas[1]},
                        {'params': alphas[2]}
                    ], lr=3.0, momentum=0.9, maximize=True)
                    assert len(alphas) == 3  # othwerwise, we probably need to optimize them as well
                    if direction == "lbs" and (branch.resulting_lbs[layeri][neuron] >= 0.0) and layeri > 0: continue
                    if direction == "ubs" and (branch.resulting_ubs[layeri][neuron] <= 0.0) and layeri > 0: continue
                    for _ in range(10):
                        optim.zero_grad()
                        loss = optimize_bound(branch.weights, branch.biases, gamma, alphas, branch.resulting_lbs, branch.resulting_ubs, num_layers, layeri, neuron, direction)
                        loss.backward()
                        optim.step()

                        with torch.no_grad():
                            if direction == "lbs":
                                branch.resulting_lbs[layeri][neuron] = torch.max(branch.resulting_lbs[layeri][neuron], loss.detach())
                            else:
                                branch.resulting_ubs[layeri][neuron] = torch.min(branch.resulting_ubs[layeri][neuron], -loss.detach())
                            if branch.resulting_lbs[layeri][neuron] > branch.resulting_ubs[layeri][neuron]:
                                tqdm.write("[WARNING] Infeasible bounds determined. That's either a bug, or this input region has no intersection with the target area")
                                abort = True
                                break
                            gamma.data = torch.clamp(gamma.data, min=0)
                            alphas[1].data = alphas[1].data.clamp(min=0.0, max=1.0)
                            alphas[2].data = alphas[2].data.clamp(min=0.0, max=1.0)

            if abort:
                break
            m, xs, zs = get_triangle_grb_model(model, branch.resulting_ubs, branch.resulting_lbs, H, d, input_lbs, input_ubs)
                
            gurobi_infeasible_counter = 0
            for i, c in tqdm(enumerate(cs), desc="cs", leave=False):
                b = get_optimized_grb_result(m, c, zs[0])
                if not b:
                    gurobi_infeasible_counter += 1
                    continue
                if i == 0:
                    last_b = b
                pending_approximated_input_bounds.append(ApproximatedInputBound(branch.input_lbs, branch.input_ubs, c, b))
            if gurobi_infeasible_counter > 0:
                tqdm.write("[WARNING] Gurobi determined that the bounds are infeasible. That's either a bug or this input region as no intersection with the target area")
                assert gurobi_infeasible_counter == len(cs)
                abort = True
                break

            plot2d(model, H, d, approximated_input_bounds + pending_approximated_input_bounds, input_lbs, input_ubs, plot_number=plot_number, save=True, branch=branch, contour=contour)
            plot_number += 1
        approximated_input_bounds += pending_approximated_input_bounds
    plot2d(model, H, d, approximated_input_bounds, input_lbs, input_ubs, plot_number=plot_number, save=True, contour=contour)
    input("Press enter to terminate")
