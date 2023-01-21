from typing import List, Optional, Tuple

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

def _get_initial_input_branch(
    model,
    H,
    d,
    cs: torch.Tensor,  # (num_cs, featInputLayer)
    input_lbs,
    input_ubs,
    max_branching_depth: Optional[int],
    initial_resulting_lbs: Optional[List[Optional[torch.Tensor]]],  # (feat)
    initial_resulting_ubs: Optional[List[Optional[torch.Tensor]]],  # (feat)
):
    (resulting_lbs, resulting_ubs), (cs_lbs, cs_ubs), params_dict, weights, biases = initialize_all(
        model=model,
        input_lbs=torch.Tensor(input_lbs),
        input_ubs=torch.Tensor(input_ubs),
        H=H,
        d=d,
        cs=cs,
        initial_resulting_lbs=initial_resulting_lbs,
        initial_resulting_ubs=initial_resulting_ubs,
    )
    initial_input_branch = InputBranch(
        input_lbs=torch.Tensor(input_lbs),
        input_ubs=torch.Tensor(input_ubs),
        params_dict=params_dict,
        resulting_lbs=resulting_lbs,
        resulting_ubs=resulting_ubs,
        cs=cs,
        cs_lbs=cs_lbs,
        cs_ubs=cs_ubs,
        weights=weights,
        biases=biases,
        remaining_max_branching_depth=max_branching_depth,
    )
    return initial_input_branch

def optimize(
    model,
    H,
    d,
    cs: torch.Tensor,  # (num_cs, featInputLayer)
    input_lbs,
    input_ubs,
    max_num_iters,
    convergence_threshold: float,
    max_branching_depth: Optional[int],
    plotting_level: PlottingLevel,
    load_bounds_of_stacked: Optional[int] = None,
    save_bounds_as_stacked: Optional[int] = None,
    dont_optimize_loaded_layers: bool = False,
):
    plt.ion()
    plt.show()

    # Output the Gurobi-Text now
    gp.Model()

    # very small values can cause numerical instability/inf/nans during plotting
    cs = torch.where(torch.abs(cs) > 0.0001, cs, 0.0001)

    approximated_input_bounds: List[ApproximatedInputBound] = []
    excluded_input_regions: List[ExcludedInputRegions] = []

    num_layers = get_num_layers(model)
    initial_resulting_lbs: List[Optional[torch.Tensor]] = [None] * num_layers  # (feat)
    initial_resulting_ubs: List[Optional[torch.Tensor]] = [None] * num_layers  # (feat)
    optmize_layer: List[bool] = [True] * num_layers
    if load_bounds_of_stacked is not None:
        loaded_lbs = np.load(f"resulting_lbs{load_bounds_of_stacked}.npy", allow_pickle=True)
        loaded_ubs = np.load(f"resulting_ubs{load_bounds_of_stacked}.npy", allow_pickle=True)
        assert len(loaded_lbs) == len(loaded_ubs)
        for i in range(1, len(loaded_lbs)):
            initial_resulting_lbs[-i] = loaded_lbs[-i]
            initial_resulting_ubs[-i] = loaded_ubs[-i]
            if dont_optimize_loaded_layers:
                optmize_layer[-i] = False

    branches = [_get_initial_input_branch(
        model=model,
        H=H,
        d=d,
        cs=cs,
        input_lbs=input_lbs,
        input_ubs=input_ubs,
        max_branching_depth=max_branching_depth,
        initial_resulting_lbs=initial_resulting_lbs,
        initial_resulting_ubs=initial_resulting_ubs,
    )]

    plot_number = 0
    root_branch_resulting_lbs = None
    root_branch_resulting_ubs = None

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
        for iteration in pbar:
            if branch_excluded or branch_converged:
                break
            pending_approximated_input_bounds = []
            for inner_iteration in tqdm(range(10), desc="Rounds", leave=False):
                if branch_excluded:
                    break
                for layeri in tqdm(get_layer_indices(model), desc="Layers", leave=False):
                    if not optmize_layer[layeri]:
                        continue

                    if branch_excluded:
                        break
                    # batch size = features in layer i (+ num_cs in the input layer)

                    # we need to make sure that the bounds on cs_bounds are as tight as possible
                    # therefore, we spend extra time on those bounds
                    if layeri == 0 and inner_iteration == 9:
                        r = 10
                    else:
                        r = 1
                    for _ in range(r):
                        gamma = branch.params_dict[layeri]['gamma']  # (2, batch, 1, 1)
                        alphas = branch.params_dict[layeri]['alphas']  # [(2, batch, feat)]
                        optim = branch.optimizers[layeri]

                        optim.zero_grad(set_to_none=True)
                        loss = optimize_bound(branch.weights, branch.biases, gamma, alphas, branch.resulting_lbs, branch.resulting_ubs, num_layers, layeri, cs)  # (dir==2, batch, 1)
                        assert loss.dim() == 3, loss.shape
                        assert loss.size(0) == 2
                        if layeri == 0:
                            assert loss.size(1) == get_num_neurons(model, layeri) + cs.size(0)
                        else:
                            assert loss.size(1) == get_num_neurons(model, layeri)
                        assert loss.size(2) == 1
                        optimized_bounds = loss.detach().squeeze(dim=-1)  # (dir==2, batch)
                        loss = loss.sum()
                        loss.backward()
                        optim.step()

                        with torch.no_grad():
                            if layeri == 0:
                                num_input_features = get_num_neurons(model, 0)
                                layer_bounds = optimized_bounds[:, :num_input_features]
                                cs_bounds = optimized_bounds[:, num_input_features:]
                                assert cs_bounds.size(1) == cs.size(0)
                                branch.cs_lbs = torch.max(branch.cs_lbs, cs_bounds[0])
                                branch.cs_ubs = torch.min(branch.cs_ubs, -cs_bounds[1])
                            else:
                                layer_bounds = optimized_bounds

                            branch.resulting_lbs[layeri] = torch.max(branch.resulting_lbs[layeri], layer_bounds[0])
                            branch.resulting_ubs[layeri] = torch.min(branch.resulting_ubs[layeri], -layer_bounds[1])
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
                
            b_sum = 0
            for i, c in tqdm(enumerate(cs), desc="cs", leave=False):
                pending_approximated_input_bounds.append(ApproximatedInputBound(branch.input_lbs.cpu(), branch.input_ubs.cpu(), c.cpu().numpy(), branch.cs_lbs[i].cpu().numpy()))
                pending_approximated_input_bounds.append(ApproximatedInputBound(branch.input_lbs.cpu(), branch.input_ubs.cpu(), -c.cpu().numpy(), -branch.cs_ubs[i].cpu().numpy()))
                b_sum += branch.cs_lbs[i]
                b_sum += -branch.cs_ubs[i]
            if branch.last_b_sum is None:
                branch.last_b_sum = b_sum
            if convergence_threshold is not None:
                if b_sum > branch.last_b_sum + convergence_threshold * 2 * len(cs):
                    b_sum_improved_once = True
                elif b_sum_improved_once:
                    branch_converged = True
            branch.last_b_sum = b_sum

            if plotting_level == PlottingLevel.ALL_STEPS:
                plot2d(model, H, d, approximated_input_bounds + pending_approximated_input_bounds, excluded_input_regions, input_lbs, input_ubs, plot_number=plot_number, save=True, branch=branch, contour=False)
                plot_number += 1

        if root_branch_resulting_lbs is None:
            assert root_branch_resulting_ubs is None
            root_branch_resulting_lbs = deepcopy(branch.resulting_lbs)
            root_branch_resulting_ubs = deepcopy(branch.resulting_ubs)

        if branch_excluded:
            excluded_input_regions.append(ExcludedInputRegions(branch.input_lbs.cpu(), branch.input_ubs.cpu()))
        else:
            approximated_input_bounds += pending_approximated_input_bounds
            if branch.remaining_max_branching_depth is None or branch.remaining_max_branching_depth > 0:
                branches += branch.split()
    if plotting_level in [PlottingLevel.ALL_STEPS, PlottingLevel.FINAL_ONLY]:
        plot2d(model, H, d, approximated_input_bounds, excluded_input_regions, input_lbs, input_ubs, plot_number=plot_number, save=True, contour=False)
        input("Press enter to terminate")

    if save_bounds_as_stacked is not None:
        np.save(f"resulting_lbs{save_bounds_as_stacked}.npy", root_branch_resulting_lbs)
        np.save(f"resulting_ubs{save_bounds_as_stacked}.npy", root_branch_resulting_ubs)
