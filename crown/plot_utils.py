from enum import Enum
from typing import List, Optional

import numpy as np
import gurobipy as gp
import math
import torch
from torch.autograd import Variable

import core.trainer as trainer
import matplotlib.pyplot as plt

from crown.approx_utils import get_remaining_input_area_mask

from .branch_utils import ApproximatedInputBound, ExcludedInputRegions, InputBranch
from .model_utils import get_num_layers

class PlottingLevel(Enum):
    NO_PLOTTING = 1
    FINAL_ONLY = 2
    ALL_STEPS = 3

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
    contour: bool = True,
):
    # plt.rcParams["figure.figsize"] = (6, 3.6)
    plt.rcParams["figure.figsize"] = (6, 6)
    # plt.cla()

    MIN_X_INPUT_VALUE = input_lbs[0]
    MIN_Y_INPUT_VALUE = input_lbs[1]
    MAX_X_INPUT_VALUE = input_ubs[0]
    MAX_Y_INPUT_VALUE = input_ubs[1]

    assert len(input_lbs) == 2 and len(input_ubs) == 2

    resolution_x = 1000
    resolution_y = 1000
    
    XX, YY = np.meshgrid(np.linspace(MIN_X_INPUT_VALUE, MAX_X_INPUT_VALUE, resolution_x), np.linspace(MIN_Y_INPUT_VALUE, MAX_Y_INPUT_VALUE, resolution_y))
    X0 = Variable(torch.tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T, device="cpu", dtype=torch.float32))
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently#comment104116008_58926343
    orig_model_location = next(model.parameters()).device
    model.to("cpu")
    y0 = model(X0)
    model.to(orig_model_location)
    output_constraints = torch.all(H.cpu().matmul(y0.unsqueeze(-1)).squeeze(-1) + d.cpu() <= 0, dim=1)
    target_area = (output_constraints).resize(resolution_y,resolution_x).data.numpy()

    # Uncomment to plot the area specified by H and d
    # obstacle = np.all(X0.cpu().numpy() @ H.cpu().numpy().T + d.cpu().numpy() <= 0, axis=1)
    # obstacle.resize(resolution_y,resolution_x)
    # plt.contour(XX,YY,obstacle, colors="red", levels=[0,1])

    if contour:
        id = torch.max(y0[:,0], y0[:,1])
        ZZ = (y0[:,2] - id).resize(resolution_y,resolution_x).data.numpy()
        bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1
        plt.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))

    plt.contourf(XX,YY,target_area, alpha=1.0, levels=[0.5, 1.5], extend="neither", color="green")

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

    def plot_box(x1, y1, x2, y2, color="red"):
        plt.plot(np.array([x1, x2, x2, x1, x1]), np.array([y1, y1, y2, y2, y1]), color=color)

    def fill_box(x1, y1, x2, y2, color="red"):
        plt.fill(np.array([x1, x2, x2, x1, x1]), np.array([y1, y1, y2, y2, y1]), color=color)

    for approximated_input_bound in approximated_input_bounds:
        c = approximated_input_bound.c
        b = approximated_input_bound.b
        x_vals = np.array([approximated_input_bound.input_lbs[0], approximated_input_bound.input_ubs[0]])
        y_vals = np.array([approximated_input_bound.input_lbs[1], approximated_input_bound.input_ubs[1]])
        from copy import deepcopy
        slope =  -c[0] / c[1]
        intercept = b / c[1]
        abline(deepcopy(x_vals), deepcopy(y_vals), slope, intercept)
    for excluded_input_region in excluded_input_regions:
        plot_box(excluded_input_region.input_lbs[0].cpu(), excluded_input_region.input_lbs[1].cpu(), excluded_input_region.input_ubs[0].cpu(), excluded_input_region.input_ubs[1].cpu())


    if branch is not None:
        plt.plot(np.array([branch.input_lbs[0].cpu(), branch.input_ubs[0].cpu(), branch.input_ubs[0].cpu(), branch.input_lbs[0].cpu(), branch.input_lbs[0].cpu()]),
                 np.array([branch.input_lbs[1].cpu(), branch.input_lbs[1].cpu(), branch.input_ubs[1].cpu(), branch.input_ubs[1].cpu(), branch.input_lbs[1].cpu()]), color="black")

    remaining_input_area = get_remaining_input_area_mask(
        min_x_input_value=MIN_X_INPUT_VALUE,
        max_x_input_value=MAX_X_INPUT_VALUE,
        min_y_input_value=MIN_Y_INPUT_VALUE,
        max_y_input_value=MAX_Y_INPUT_VALUE,
        approximated_input_bounds=approximated_input_bounds,
        excluded_input_regions=excluded_input_regions
    )
    plt.contour(XX, YY, 1 - remaining_input_area, colors="blue", levels=[0,1])

    plt.title(f"Step {plot_number+1}")
    
    # STYLE 2: SAVING INPUT AREAS
    # for num_layers in range(5, 45, 4):
    #     remaining_input_area = np.load(f"remaining_input_area{num_layers}.npy", allow_pickle=True)
    #     plt.contour(XX, YY, 1 - remaining_input_area, colors="blue", levels=[0,1])
    # np.save(f"remaining_input_area{get_num_layers(model)}.npy", remaining_input_area)
    
    # STYLE 3: BOXES: 
    # boxes = [
    # [[3.581497320944417, 4.534045635948357], [1.066834716061575, 1.6510319172051053]]
    # ,[[1.7475117321061344, 2.8503394963378526], [2.1017427525322323, 2.332267753666301]]
    # ,[[-0.292872522650807, 0.7048653553123362], [1.6666809835259901, 2.1581685810254525]]
    # ,[[-1.9076967547687556, -0.6819325125905512], [0.9110064467608845, 1.3446047950182334]]
    # ,[[-2.778093782362992, -1.341332254242789], [0.18662401285962185, 0.6183378346605897]]
    # ,[[-3.009010513421755, -1.3876906578882762], [-0.36338474466649595, 0.10407833362225868]]
    # ,[[-2.830935133450058, -0.8804047850522396], [-0.7278299027257042, 0.12219277840838015]]
    # ,[[-2.6884987157336973, 0.11675366257062164], [-1.266486992541268, 0.15042944936085195]]
    # ,[[-2.663876858549674, 2.0499154497078997], [-2.5998365817524745, 0.5271764942559776]]
    # ,[[-2.8915754368444584, 4.999999998154876], [-4.9999999998598685, 0.758374639111246]]
    # ]
    # for box in boxes:
    #     plot_box(box[0][0], box[1][0], box[0][1], box[1][1])

    plt.draw()
    plt.pause(1)

    if save:
        plt.savefig(f"plots/step{plot_number+1}.png", dpi=1200)