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

from .model_utils import get_num_layers, get_num_neurons, get_layer_indices
from .lp import get_optimized_grb_result, get_triangle_grb_model
from .crown import initialize_all, initialize_params, optimize_bound, initialize_bounds, tighten_bounds_with_rsip
from .plot_utils import plot2d
from .branch_utils import InputBranch, ApproximatedInputBound, ExcludedInputRegions

def optimize(model, H, d, num_cs, input_lbs, input_ubs, num_iters, max_branching_depth=None, contour=True, verbose_plotting=False):
    plt.ion()
    plt.show()

    # Output the Gurobi-Text now
    gp.Model()

    # based on https://math.stackexchange.com/a/4388888/50742
    cs = [[np.cos(2*np.pi*t / num_cs), np.sin(2*np.pi*t / num_cs)] for t in range(num_cs)]
    for i, c in enumerate(cs):
        # very small values can cause numerical instability/inf/nans during plotting
        if np.abs(c[0]) < 0.0001:
            cs[i][0] = 0.0001
        if np.abs(c[1]) < 0.0001:
            cs[i][1] = 0.0001

    approximated_input_bounds: List[ApproximatedInputBound] = []
    excluded_input_regions: List[ExcludedInputRegions] = []


    def get_initial_input_branch(model, H, d):
        resulting_lbs, resulting_ubs, params_dict, weights, biases = initialize_all(
            model=model,
            input_lbs=torch.Tensor(input_lbs),
            input_ubs=torch.Tensor(input_ubs),
            H=H,
            d=d
        )
        initial_input_branch = InputBranch(
            input_lbs=torch.Tensor(input_lbs),
            input_ubs=torch.Tensor(input_ubs),
            params_dict=params_dict,
            resulting_lbs=resulting_lbs,
            resulting_ubs=resulting_ubs,
            weights=weights,
            biases=biases,
            remaining_max_branching_depth=max_branching_depth)
        return initial_input_branch

    branches = [get_initial_input_branch(model, H, d)]

    plot_number = 0
    num_layers = get_num_layers(model)
    while True:
        if len(branches) == 0:
            break
        branch = branches[0]
        branches = branches[1:]

        abort = False
        for layeri in range(get_num_layers(model)):
            if torch.any(branch.resulting_lbs[layeri] > branch.resulting_ubs[layeri]):
                tqdm.write(f"[WARNING] Next branch has invalid bounds at layer {layeri}. That's either a bug, or this input region has no intersection with the target area")
                abort = True  # we'll skip the optimizations and jump to the end of this loop where the current input region is marked as unreachable


        gamma_global, alphas_global = initialize_params(branch.weights, num_layers, 2) 
        optim = torch.optim.SGD([{'params': gamma_global, 'lr' : 0.001}, {'params': alphas_global[1:]}], lr=3.0, momentum=0.9, maximize=True)
        for layeri in get_layer_indices(model):
            gamma = branch.params_dict[layeri]['gamma']  # (dir==2, batch, 1, 1)
            alphas = branch.params_dict[layeri]['alphas']  # [(dir==2, feat, feat)]
            optim.add_param_group({'params': gamma, 'lr' : 0.001})
            optim.add_param_group({'params': alphas[1:]})
            
        pbar = tqdm(range(num_iters), leave=False)
        last_b = []
        pending_approximated_input_bounds: List[ApproximatedInputBound] = []
        for _ in pbar:
            if abort:
                break
            pending_approximated_input_bounds = []
            pbar.set_description(f"Best solution to first bound: {last_b}")

            for iteration in range(10 * len(get_layer_indices(model))):
                optim.zero_grad(set_to_none=True)

                if iteration > 0:
                    for layeri in tqdm(get_layer_indices(model), desc="Layers", leave=False):
                        if abort:
                            break
                        # batch size = features in layer i

                        gamma = branch.params_dict[layeri]['gamma']  # (2, batch, 1, 1)
                        alphas = branch.params_dict[layeri]['alphas']  # [(2, batch, feat)]

                        optimized_bounds = optimize_bound(branch.weights, branch.biases, gamma, alphas, branch.resulting_lbs, branch.resulting_ubs, num_layers, layeri)  # (dir==2, batch, 1)
                        assert optimized_bounds.dim() == 3, optimized_bounds.shape
                        assert optimized_bounds.size(0) == 2
                        assert optimized_bounds.size(1) == get_num_neurons(model, layeri)
                        assert optimized_bounds.size(2) == 1

                        branch.resulting_lbs[layeri] = torch.max(branch.resulting_lbs[layeri], optimized_bounds[0].squeeze(dim=1))
                        branch.resulting_ubs[layeri] = torch.min(branch.resulting_ubs[layeri], -optimized_bounds[1].squeeze(dim=1))
                        if torch.any(branch.resulting_lbs[layeri] > branch.resulting_ubs[layeri]):
                            tqdm.write("[WARNING] Infeasible bounds determined. That's either a bug, or this input region has no intersection with the target area")
                            abort = True
                            break
                loss = optimize_bound(branch.weights, branch.biases, gamma_global, alphas_global, branch.resulting_lbs, branch.resulting_ubs, num_layers, 0)  # (2, batch, 1)
                loss = loss.sum()
                loss.backward(retain_graph=True)
                optim.step()

                for layeri in get_layer_indices(model):
                    gamma = branch.params_dict[layeri]['gamma']  # (dir==2, batch, 1, 1)
                    alphas = branch.params_dict[layeri]['alphas']  # [(dir==2, feat, feat)]
                    gamma.data = torch.clamp(gamma.data, min=0)
                    for i in range(1, len(alphas)):
                        alphas[i].data = alphas[i].data.clamp(min=0.0, max=1.0)
                gamma_global.data = torch.clamp(gamma_global.data, min=0)
                for i in range(1, len(alphas_global)):
                    alphas_global[i].data = alphas_global[i].data.clamp(min=0.0, max=1.0)
                # for layeri in tqdm(get_layer_indices(model), desc="Layers", leave=False):
                #     branch.resulting_lbs[layeri] = branch.resulting_lbs[layeri].detach()
                #     branch.resulting_ubs[layeri] = branch.resulting_ubs[layeri].detach()

            # with torch.no_grad():
            branch.resulting_lbs, branch.resulting_ubs = tighten_bounds_with_rsip(num_layers, branch.weights, branch.biases, branch.input_lbs, branch.input_ubs, initial_lbs=branch.resulting_lbs, initial_ubs=branch.resulting_ubs, alphas=None)
            for l, u in zip(branch.resulting_lbs, branch.resulting_ubs):
                if not torch.all(l <= u):
                    abort = True
                    break

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
                pending_approximated_input_bounds.append(ApproximatedInputBound(branch.input_lbs.cpu(), branch.input_ubs.cpu(), c, b))
            if gurobi_infeasible_counter > 0:
                tqdm.write("[WARNING] Gurobi determined that the bounds are infeasible. That's either a bug or this input region as no intersection with the target area")
                assert gurobi_infeasible_counter == len(cs)
                abort = True
                break

            if verbose_plotting:
                plot2d(model, H, d, approximated_input_bounds + pending_approximated_input_bounds, excluded_input_regions, input_lbs, input_ubs, plot_number=plot_number, save=True, branch=branch, contour=contour)
                plot_number += 1
        if abort:
            excluded_input_regions.append(ExcludedInputRegions(branch.input_lbs.cpu(), branch.input_ubs.cpu()))
        else:
            approximated_input_bounds += pending_approximated_input_bounds
            if branch.remaining_max_branching_depth is None or branch.remaining_max_branching_depth > 0:
                branches += branch.split()
    plot2d(model, H, d, approximated_input_bounds, excluded_input_regions, input_lbs, input_ubs, plot_number=plot_number, save=True, contour=contour)
    input("Press enter to terminate")
