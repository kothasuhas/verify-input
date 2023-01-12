from typing import List, Optional

import numpy as np
import gurobipy as gp
import math
import torch
from torch.autograd import Variable

import core.trainer as trainer
import matplotlib.pyplot as plt

from crown.crown import ApproximatedInputBound, ExcludedInputRegions, InputBranch

def plot2d(
    model: trainer.nn.Sequential,
    H: torch.Tensor,
    d: torch.Tensor,
    approximated_input_bounds: List[ApproximatedInputBound],
    excluded_input_regions: List[ExcludedInputRegions],
    input_lbs: List[torch.Tensor],
    input_ubs: List[torch.Tensor],
    plot_number: int,
    save: bool,
    branch: Optional[InputBranch] = None,
    contour: bool =True
):
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.cla()

    MIN_X_INPUT_VALUE = input_lbs[0]
    MIN_Y_INPUT_VALUE = input_lbs[1]
    MAX_X_INPUT_VALUE = input_ubs[0]
    MAX_Y_INPUT_VALUE = input_ubs[1]

    assert len(input_lbs) == 2 and len(input_ubs) == 2

    resolution_x = 1000
    resolution_y = 1000
    remaining_input_area = np.ones((resolution_y, resolution_x))
    XX, YY = np.meshgrid(np.linspace(MIN_X_INPUT_VALUE, MAX_X_INPUT_VALUE, resolution_x), np.linspace(MIN_Y_INPUT_VALUE, MAX_Y_INPUT_VALUE, resolution_y))
    X0 = Variable(torch.tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T, device="cpu", dtype=torch.float32))
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently#comment104116008_58926343
    orig_model_location = next(model.parameters()).device
    model.to("cpu")
    y0 = model(X0)
    model.to(orig_model_location)
    output_constraints = torch.all(H.cpu().matmul(y0.unsqueeze(-1)).squeeze(-1) + d.cpu() <= 0, dim=1)
    target_area = (output_constraints).resize(resolution_y,resolution_x).data.numpy()

    plt.contour(XX,YY,target_area, colors="green", levels=[0,1])
    if contour:
        id = torch.max(y0[:,0], y0[:,1])
        ZZ = (y0[:,2] - id).resize(resolution_y,resolution_x).data.numpy()
        bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1
        plt.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))

    plt.axis("equal")

    plt.xlim(MIN_X_INPUT_VALUE - 0.1, MAX_X_INPUT_VALUE + 0.1)
    plt.ylim(MIN_Y_INPUT_VALUE - 0.1, MAX_Y_INPUT_VALUE + 0.1)


    # t = np.linspace(0, 2 * math.pi, resolution_x)
    # radius = 0.5
    # plt.plot(-1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")
    # plt.plot( 1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")

    def abline(x_vals, asserted_y_vals, slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        y_vals = intercept + slope * x_vals

        min_y_val = min(y_vals)
        max_y_val = max(y_vals)
        min_asserted_y_val = asserted_y_vals[0]
        max_asserted_y_val = asserted_y_vals[1]
        if slope < 0:
            if max_y_val > max_asserted_y_val: # a
                new_min_x_val = (max_asserted_y_val - intercept) / slope
                assert new_min_x_val > x_vals[0]
                # plt.plot(np.array([x_vals[0], new_min_x_val]), np.array([max_asserted_y_val, max_asserted_y_val]), '--', color="black")
                x_vals[0] = min(new_min_x_val, x_vals[1])
                # print("1a" f"{x_vals=}", f"{y_vals=}", f"{asserted_y_vals=}")
            else:
                assert max_asserted_y_val >= max_y_val # b
                # print("1b")
                # plt.plot(np.array([x_vals[0], x_vals[0]]), np.array([max_y_val, max_asserted_y_val]), '--', color="black")
            
            if min_y_val < min_asserted_y_val: # c
                new_max_x_val = (min_asserted_y_val - intercept) / slope
                assert new_max_x_val < x_vals[1]
                # plt.plot(np.array([new_max_x_val, x_vals[1]]), np.array([min_asserted_y_val, min_asserted_y_val]), '--', color="black")
                x_vals[1] = max(new_max_x_val, x_vals[0])
                # print("1c", f"{x_vals=}", f"{y_vals=}", f"{asserted_y_vals=}")
            else:
                assert min_asserted_y_val <= min_y_val
                # print("1d")
                # plt.plot(np.array([x_vals[1], x_vals[1]]), np.array([min_asserted_y_val, min_y_val]), '--', color="black")
        else:
            assert slope > 0
            if max_y_val > max_asserted_y_val: # a
                new_max_x_val = (max_asserted_y_val - intercept) / slope
                assert new_max_x_val < x_vals[1]
                # plt.plot(np.array([new_max_x_val, x_vals[1]]), np.array([max_asserted_y_val, max_asserted_y_val]), '--', color="black")
                x_vals[1] = max(new_max_x_val, x_vals[0])
                # print("2a" f"{x_vals=}", f"{y_vals=}", f"{asserted_y_vals=}")
            else:
                assert max_asserted_y_val >= max_y_val # b
                # print("2b")
                # plt.plot(np.array([x_vals[1], x_vals[1]]), np.array([max_asserted_y_val, max_y_val]), '--', color="black")
            
            if min_y_val < min_asserted_y_val: # c
                new_min_x_val = (min_asserted_y_val - intercept) / slope
                assert new_min_x_val > x_vals[0]
                # plt.plot(np.array([x_vals[0], new_min_x_val]), np.array([min_asserted_y_val, min_asserted_y_val]), '--', color="black")
                x_vals[0] = min(new_min_x_val, x_vals[1])
                # print("2c" f"{x_vals=}", f"{y_vals=}", f"{asserted_y_vals=}")
            else:
                assert min_asserted_y_val <= min_y_val # d
                # print("2d")
                # plt.plot(np.array([x_vals[0], x_vals[0]]), np.array([min_asserted_y_val, min_y_val]), '--', color="black")
             

        y_vals = intercept + slope * x_vals
        plt.fill(x_vals, y_vals, '--', color="red")

    def get_branch_input_area_mask(input_lbs: List[torch.Tensor], input_ubs: List[torch.Tensor]):
        ticks_per_width_x = resolution_x / (MAX_X_INPUT_VALUE - MIN_X_INPUT_VALUE)
        ticks_per_width_y = resolution_y / (MAX_Y_INPUT_VALUE - MIN_Y_INPUT_VALUE)
        branch_input_area_mask = np.zeros((resolution_y, resolution_x), dtype=np.bool8)
        branch_x_lbs_in_ticks = math.ceil((input_lbs[0] - MIN_X_INPUT_VALUE) * ticks_per_width_x)
        branch_x_ubs_in_ticks = math.floor((input_ubs[0] - MIN_X_INPUT_VALUE) * ticks_per_width_x)
        branch_y_lbs_in_ticks = math.ceil((input_lbs[1] - MIN_Y_INPUT_VALUE) * ticks_per_width_y)
        branch_y_ubs_in_ticks = math.floor((input_ubs[1] - MIN_Y_INPUT_VALUE) * ticks_per_width_y)
        branch_input_area_mask[branch_y_lbs_in_ticks:branch_y_ubs_in_ticks+1, branch_x_lbs_in_ticks:branch_x_ubs_in_ticks+1] = 1
        return branch_input_area_mask

    for approximated_input_bound in approximated_input_bounds:
        c = approximated_input_bound.c
        b = approximated_input_bound.b
        x_vals = np.array([approximated_input_bound.input_lbs[0], approximated_input_bound.input_ubs[0]])
        y_vals = np.array([approximated_input_bound.input_lbs[1], approximated_input_bound.input_ubs[1]])
        from copy import deepcopy
        slope =  -c[0] / c[1]
        intercept = b / c[1]
        abline(deepcopy(x_vals), deepcopy(y_vals), slope, intercept)

        branch_input_area_mask = get_branch_input_area_mask(approximated_input_bound.input_lbs, approximated_input_bound.input_ubs)

        excluded_halfspace = np.zeros((resolution_y, resolution_x), dtype=np.bool8)
        if c[1] > 0:
            excluded_halfspace[intercept + XX * slope > YY] = 1
        else:
            assert c[1] < 0
            excluded_halfspace[intercept + XX * slope < YY] = 1

        remaining_input_area[branch_input_area_mask * excluded_halfspace] = 0

    for excluded_input_region in excluded_input_regions:
        branch_input_area_mask = get_branch_input_area_mask(excluded_input_region.input_lbs, excluded_input_region.input_ubs)
        remaining_input_area[branch_input_area_mask] = 0
        plt.plot(np.array([excluded_input_region.input_lbs[0].cpu(), excluded_input_region.input_ubs[0].cpu(), excluded_input_region.input_ubs[0].cpu(), excluded_input_region.input_lbs[0].cpu(), excluded_input_region.input_lbs[0].cpu()]),
                 np.array([excluded_input_region.input_lbs[1].cpu(), excluded_input_region.input_lbs[1].cpu(), excluded_input_region.input_ubs[1].cpu(), excluded_input_region.input_ubs[1].cpu(), excluded_input_region.input_lbs[1].cpu()]), color="red")


    if branch is not None:
        plt.plot(np.array([branch.input_lbs[0].cpu(), branch.input_ubs[0].cpu(), branch.input_ubs[0].cpu(), branch.input_lbs[0].cpu(), branch.input_lbs[0].cpu()]),
                 np.array([branch.input_lbs[1].cpu(), branch.input_lbs[1].cpu(), branch.input_ubs[1].cpu(), branch.input_ubs[1].cpu(), branch.input_lbs[1].cpu()]), color="black")

    plt.contourf(XX, YY, remaining_input_area, hatches=["xxxx", ""], alpha=0, levels=[0, 0.5])

    plt.title(f"Step {plot_number+1}")

    plt.draw()
    plt.pause(1)

    if save:
        plt.savefig(f"plots/step{plot_number+1}.png")