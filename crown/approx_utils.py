import math
from typing import List
import numpy as np
import torch

from crown.branch_utils import ApproximatedInputBound, ExcludedInputRegions


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
    branch_input_area_mask[branch_y_lbs_in_ticks:branch_y_ubs_in_ticks+1, branch_x_lbs_in_ticks:branch_x_ubs_in_ticks+1] = 1
    return branch_input_area_mask

def get_remaining_input_area_mask(
    min_x_input_value: int,
    max_x_input_value: int,
    min_y_input_value: int,
    max_y_input_value: int,
    approximated_input_bounds: List[ApproximatedInputBound],
    excluded_input_regions: List[ExcludedInputRegions],
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
            assert c[1] < 0
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