from typing import List, Optional

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
from .crown import initialize_all, optimize_bound, initialize_bounds, tighten_bounds_with_rsip
from .plot_utils import PlottingLevel, plot2d
from .branch_utils import InputBranch, ApproximatedInputBound, ExcludedInputRegions

def optimize(
    model,
    H,
    d,
    cs,
    input_lbs,
    input_ubs,
    max_num_iters,
    convergence_threshold: float,
    max_branching_depth: Optional[int],
    plotting_level: PlottingLevel
):
    plt.ion()
    plt.show()

    # Output the Gurobi-Text now
    gp.Model()

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

        branch_excluded = False
        branch_converged = False
        for layeri in range(get_num_layers(model)):
            if torch.any(branch.resulting_lbs[layeri] > branch.resulting_ubs[layeri]):
                tqdm.write(f"[WARNING] Next branch has invalid bounds at layer {layeri}. That's either a bug, or this input region has no intersection with the target area")
                branch_excluded = True  # we'll skip the optimizations and jump to the end of this loop where the current input region is marked as unreachable

        pbar = tqdm(range(max_num_iters), leave=False)
        b_sum_improved_once = False
        pending_approximated_input_bounds: List[ApproximatedInputBound] = []
        for _ in pbar:
            if branch_excluded or branch_converged:
                break
            pending_approximated_input_bounds = []
            pbar.set_description(f"Sum of best solutions: {branch.last_b_sum}")
            for layeri in tqdm(get_layer_indices(model), desc="Layers", leave=False):
                if branch_excluded:
                    break
                # batch size = features in layer i

                gamma = branch.params_dict[layeri]['gamma']  # (2, batch, 1, 1)
                alphas = branch.params_dict[layeri]['alphas']  # [(2, batch, feat)]
                optim = branch.optimizers[layeri]

                for _ in range(10):
                    optim.zero_grad(set_to_none=True)
                    loss = optimize_bound(branch.weights, branch.biases, gamma, alphas, branch.resulting_lbs, branch.resulting_ubs, num_layers, layeri)  # (dir==2, batch, 1)
                    assert loss.dim() == 3, loss.shape
                    assert loss.size(0) == 2
                    assert loss.size(1) == get_num_neurons(model, layeri)
                    assert loss.size(2) == 1
                    optimized_bounds = loss.detach().squeeze(dim=-1)  # (dir==2, batch)
                    loss = loss.sum()
                    loss.backward()
                    optim.step()

                    with torch.no_grad():
                        branch.resulting_lbs[layeri] = torch.max(branch.resulting_lbs[layeri], optimized_bounds[0])
                        branch.resulting_ubs[layeri] = torch.min(branch.resulting_ubs[layeri], -optimized_bounds[1])
                        if torch.any(branch.resulting_lbs[layeri] > branch.resulting_ubs[layeri]):
                            tqdm.write("[WARNING] Infeasible bounds determined. That's either a bug, or this input region has no intersection with the target area")
                            branch_excluded = True
                            break
                        gamma.data = torch.clamp(gamma.data, min=0)
                        for i in range(1, len(alphas)):
                            alphas[i].data = alphas[i].data.clamp(min=0.0, max=1.0)
            branch.resulting_lbs, branch.resulting_ubs = tighten_bounds_with_rsip(num_layers, branch.weights, branch.biases, branch.input_lbs, branch.input_ubs, initial_lbs=branch.resulting_lbs, initial_ubs=branch.resulting_ubs, alphas=None)
            for l, u in zip(branch.resulting_lbs, branch.resulting_ubs):
                if not torch.all(l <= u):
                    branch_excluded = True
                    break

            if branch_excluded:
                break
            m, _, zs = get_triangle_grb_model(model, branch.resulting_ubs, branch.resulting_lbs, H, d, input_lbs, input_ubs)
                
            gurobi_infeasible_counter = 0
            b_sum = 0
            for i, c in tqdm(enumerate(cs), desc="cs", leave=False):
                b = get_optimized_grb_result(m, c, zs[0])
                if not b:
                    gurobi_infeasible_counter += 1
                    continue
                pending_approximated_input_bounds.append(ApproximatedInputBound(branch.input_lbs.cpu(), branch.input_ubs.cpu(), c, b))
                b_sum += b
            if gurobi_infeasible_counter > 0:
                tqdm.write("[WARNING] Gurobi determined that the bounds are infeasible. That's either a bug or this input region as no intersection with the target area")
                assert gurobi_infeasible_counter == len(cs)
                branch_excluded = True
                break
            if branch.last_b_sum is None:
                branch.last_b_sum = b_sum
            if convergence_threshold is not None:
                if b_sum > branch.last_b_sum + convergence_threshold * len(cs):
                    b_sum_improved_once = True
                elif b_sum_improved_once:
                    branch_converged = True
            branch.last_b_sum = b_sum

            if plotting_level == PlottingLevel.ALL_STEPS:
                plot2d(model, H, d, approximated_input_bounds + pending_approximated_input_bounds, excluded_input_regions, input_lbs, input_ubs, plot_number=plot_number, save=True, branch=branch, contour=False)
                plot_number += 1

        if branch_excluded:
            excluded_input_regions.append(ExcludedInputRegions(branch.input_lbs.cpu(), branch.input_ubs.cpu()))
        else:
            approximated_input_bounds += pending_approximated_input_bounds
            if branch.remaining_max_branching_depth is None or branch.remaining_max_branching_depth > 0:
                branches += branch.split()
    if plotting_level in [PlottingLevel.ALL_STEPS, PlottingLevel.FINAL_ONLY]:
        plot2d(model, H, d, approximated_input_bounds, excluded_input_regions, input_lbs, input_ubs, plot_number=plot_number, save=True, contour=False)
        input("Press enter to terminate")
