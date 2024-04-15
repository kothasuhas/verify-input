from typing import List, Optional, Tuple, Callable

import matplotlib
matplotlib.use('Svg')

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import math


def _get_branch_input_area_mask(
    resolution_x: int,
    resolution_y: int,
    min_x_input_value: int,
    max_x_input_value: int,
    min_y_input_value: int,
    max_y_input_value: int,

    input_lbs: List[torch.Tensor],
    input_ubs: List[torch.Tensor]
):

    ticks_per_width_x = resolution_x / (max_x_input_value - min_x_input_value)
    ticks_per_width_y = resolution_y / (max_y_input_value - min_y_input_value)
    branch_input_area_mask = np.zeros((resolution_y, resolution_x), dtype=np.bool8)
    branch_x_lbs_in_ticks = math.ceil((input_lbs[0] - min_x_input_value) * ticks_per_width_x)
    branch_x_ubs_in_ticks = math.floor((input_ubs[0] - min_x_input_value) * ticks_per_width_x)
    branch_y_lbs_in_ticks = math.ceil((input_lbs[1] - min_y_input_value) * ticks_per_width_y)
    branch_y_ubs_in_ticks = math.floor((input_ubs[1] - min_y_input_value) * ticks_per_width_y)
    branch_input_area_mask[branch_y_lbs_in_ticks:branch_y_ubs_in_ticks, branch_x_lbs_in_ticks:branch_x_ubs_in_ticks] = 1
    return branch_input_area_mask

def get_remaining_input_area_mask(
    min_x_input_value: int,
    max_x_input_value: int,
    min_y_input_value: int,
    max_y_input_value: int,
    approximated_input_bounds: List,
    excluded_input_regions: List = [],
):
    resolution_x = 1000
    resolution_y = 1000
    XX, YY = np.meshgrid(np.linspace(min_x_input_value, max_x_input_value, resolution_x), np.linspace(min_y_input_value, max_y_input_value, resolution_y))

    remaining_input_area = np.ones((resolution_y, resolution_x))
    for approximated_input_bound in approximated_input_bounds:
        c = approximated_input_bound.c
        b = approximated_input_bound.b
        slope =  -c[0] / c[1]
        intercept = b / c[1]

        branch_input_area_mask = _get_branch_input_area_mask(
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            min_x_input_value=min_x_input_value,
            max_x_input_value=max_x_input_value,
            min_y_input_value=min_y_input_value,
            max_y_input_value=max_y_input_value,
            input_lbs=approximated_input_bound.input_lbs,
            input_ubs=approximated_input_bound.input_ubs
        )

        excluded_halfspace = np.zeros((resolution_y, resolution_x), dtype=np.bool8)
        if c[1] > 0:
            excluded_halfspace[intercept + XX * slope > YY] = 1
        else:
            assert c[1] <= 0
            excluded_halfspace[intercept + XX * slope < YY] = 1

        remaining_input_area[branch_input_area_mask * excluded_halfspace] = 0

    for excluded_input_region in excluded_input_regions:
        branch_input_area_mask = _get_branch_input_area_mask(
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            min_x_input_value=min_x_input_value,
            max_x_input_value=max_x_input_value,
            min_y_input_value=min_y_input_value,
            max_y_input_value=max_y_input_value,
            input_lbs=excluded_input_region.input_lbs,
            input_ubs=excluded_input_region.input_ubs
        )
        remaining_input_area[branch_input_area_mask] = 0

    return remaining_input_area

def plot2d(
    model,
    H: torch.Tensor,
    d: torch.Tensor,
    constraint_merger: Callable,
    approximated_input_bounds: List,
    input_lbs: List[torch.Tensor],
    input_ubs: List[torch.Tensor],
    branch,
    name: str,
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
    
    XX, YY = np.meshgrid(np.linspace(MIN_X_INPUT_VALUE, MAX_X_INPUT_VALUE, resolution_x), np.linspace(MIN_Y_INPUT_VALUE, MAX_Y_INPUT_VALUE, resolution_y))
    X0 = Variable(torch.tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T, device=model.net.device, dtype=torch.float32))
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently#comment104116008_58926343
    y0 = model.net(X0)
    output_constraints = constraint_merger(H.cpu().matmul(y0.cpu().unsqueeze(-1)).squeeze(-1) + d.cpu() <= 0, dim=1)  # TODO: Used to be all
    target_area = (output_constraints).resize(resolution_y,resolution_x).data.numpy()
    plt.contour(XX,YY,target_area, colors="green", levels=[0,1])

    plt.axis("equal")

    plt.xlim(MIN_X_INPUT_VALUE - 0.1, MAX_X_INPUT_VALUE + 0.1)
    plt.ylim(MIN_Y_INPUT_VALUE - 0.1, MAX_Y_INPUT_VALUE + 0.1)


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
                x_vals[0] = min(new_min_x_val, x_vals[1])
            else:
                if not max_asserted_y_val >= max_y_val:
                    breakpoint() # b
            
            if min_y_val < min_asserted_y_val: # c
                new_max_x_val = (min_asserted_y_val - intercept) / slope
                assert new_max_x_val < x_vals[1]
                x_vals[1] = max(new_max_x_val, x_vals[0])
            else:
                assert min_asserted_y_val <= min_y_val
        else:
            assert slope > 0
            if max_y_val > max_asserted_y_val: # a
                new_max_x_val = (max_asserted_y_val - intercept) / slope
                assert new_max_x_val < x_vals[1]
                x_vals[1] = max(new_max_x_val, x_vals[0])
            else:
                assert max_asserted_y_val >= max_y_val # b
            
            if min_y_val < min_asserted_y_val: # c
                new_min_x_val = (min_asserted_y_val - intercept) / slope
                assert new_min_x_val > x_vals[0]
                x_vals[0] = min(new_min_x_val, x_vals[1])
            else:
                assert min_asserted_y_val <= min_y_val # d
             

        y_vals = intercept + slope * x_vals
        plt.fill(x_vals, y_vals, '--', color="red")

    for approximated_input_bound in approximated_input_bounds:
        c = approximated_input_bound.c
        b = approximated_input_bound.b
        x_vals = np.array([approximated_input_bound.input_lbs[0], approximated_input_bound.input_ubs[0]])
        y_vals = np.array([approximated_input_bound.input_lbs[1], approximated_input_bound.input_ubs[1]])
        from copy import deepcopy
        slope =  -c[0] / c[1]
        intercept = b / c[1]
        abline(deepcopy(x_vals), deepcopy(y_vals), slope, intercept)

    if branch is not None:
        plt.plot(np.array([branch.input_lbs[0].cpu(), branch.input_ubs[0].cpu(), branch.input_ubs[0].cpu(), branch.input_lbs[0].cpu(), branch.input_lbs[0].cpu()]),
                 np.array([branch.input_lbs[1].cpu(), branch.input_lbs[1].cpu(), branch.input_ubs[1].cpu(), branch.input_ubs[1].cpu(), branch.input_lbs[1].cpu()]), color="black")

    remaining_input_area = get_remaining_input_area_mask(
        min_x_input_value=MIN_X_INPUT_VALUE,
        max_x_input_value=MAX_X_INPUT_VALUE,
        min_y_input_value=MIN_Y_INPUT_VALUE,
        max_y_input_value=MAX_Y_INPUT_VALUE,
        approximated_input_bounds=approximated_input_bounds,
    )
    plt.contourf(XX, YY, remaining_input_area, hatches=["xxxx", ""], alpha=0, levels=[0, 0.5])

    plt.draw()
    plt.savefig(name)
    print(f"Result saved as {name}")

class ApproximatedInputBound:
    input_lbs: List[float]
    input_ubs: List[float]
    c: List[float]
    b: float

    def __init__(self, input_lbs, input_ubs, c, b) -> None:
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs
        self.c = c
        self.b = b
