from typing import List

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import torch
from torch.autograd import Variable
from copy import deepcopy

import core.trainer as trainer

import matplotlib.pyplot as plt


def _get_grb_model(model: trainer.nn.Sequential, layers: int):
    m = gp.Model("verify_input")
    m.Params.OutputFlag = 0

    # Create variables
    input = m.addMVar(shape=2, lb=-2, ub=2, name="input")
    xs = []
    zs = [input]
    for l in range(layers-1):
        w = model[l*2 + 1].weight.detach().numpy()
        hidden_dim = w.shape[0]
        xs.append(m.addMVar(shape=hidden_dim, lb=-1e30, ub=1e30, name=f"x{l}"))
        zs.append(m.addMVar(shape=hidden_dim, lb=-1e30, ub=1e30, name=f"z{l+1}"))
    output_dim = model[-1].weight.shape[0]
    xs.append(m.addMVar(shape=output_dim, lb=-1e30, ub=1e30, name="output"))
    return m, xs, zs

def get_num_layers(model: trainer.nn.Sequential):
    layers = len(model) // 2
    assert layers * 2 == len(model), "Model should have an even number of entries"
    return layers

def get_num_neurons(model: trainer.nn.Sequential, layer: int):
    return model[layer*2+1].weight.detach().numpy().shape[1]

def get_optimal_grb_model(model: trainer.nn.Sequential, H: torch.Tensor, d: torch.Tensor):
    layers = get_num_layers(model)

    m, xs, zs = _get_grb_model(model, layers)
    for layer in range(layers-1):
        w = model[layer*2 + 1].weight.detach().numpy()
        b = model[layer*2 + 1].bias.detach().numpy()
        hidden_dim = w.shape[0]
        m.addConstr(((w @ zs[layer]) + b) == xs[layer])
        for i in range(hidden_dim):
            m.addConstr(zs[layer+1][i] == gp.max_(xs[layer][i], constant=0))
    w = model[-1].weight.detach().numpy()
    b = model[-1].bias.detach().numpy()
    m.addConstr(((w @ zs[-1]) + b) == xs[-1])
    m.addConstr(H.detach().numpy() @ xs[-1] + d <= 0)

    m.Params.NonConvex = 2

    return m, xs, zs

def get_triangle_grb_model(model: trainer.nn.Sequential, ubs, lbs, H: torch.Tensor, d: torch.Tensor):
    layers = get_num_layers(model)
        
    m, xs, zs = _get_grb_model(model, layers)

    for layer in range(layers-1):
        w = model[layer*2 + 1].weight.detach().numpy()
        b = model[layer*2 + 1].bias.detach().numpy()
        hidden_dim = w.shape[0]
        m.addConstr(((w @ zs[layer]) + b) == xs[layer])
        for i in range(hidden_dim):
            u = ubs[layer+1][i]
            l = lbs[layer+1][i]
            assert l <= u, (l, u)
            if u <= 0:
                m.addConstr(zs[layer+1][i] == 0)
            elif l >= 0:
                m.addConstr(zs[layer+1][i] == xs[layer][i])
            else:
                m.addConstr(zs[layer+1][i] >= 0)
                m.addConstr(zs[layer+1][i] >= xs[layer][i])
                m.addConstr(zs[layer+1][i] <= xs[layer][i] * u / (u - l) - (l * u) / (u - l))
    w = model[-1].weight.detach().numpy()
    b = model[-1].bias.detach().numpy()
    m.addConstr(((w @ zs[-1]) + b) == xs[-1])
    m.addConstr(H.detach().numpy() @ xs[-1] + d.detach().numpy() <= 0)

    return m, xs, zs

def get_optimized_grb_result(m: gp.Model, c, inputs):
    m.setObjective(c[0] * inputs[0] + c[1] * inputs[1], GRB.MINIMIZE)
    m.optimize()
    if m.status != GRB.OPTIMAL:
        sc = gp.StatusConstClass
        d = {sc.__dict__[k]: k for k in sc.__dict__.keys() if k[0] >= 'A' and k[0] <= 'Z'}
        print(f"Gurobi returned a non optimal result. Error description: {d[m.status]}. Aborting.")
        exit()
                
    return m.getObjective().getValue()


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

def initialize_bounds(num_layers: int, weights: List[torch.Tensor], biases: List[torch.Tensor], input_lbs: torch.Tensor, input_ubs: torch.Tensor):
    input_lbs = deepcopy(input_lbs)
    input_ubs = deepcopy(input_ubs)

    lbs = [input_lbs]
    ubs = [input_ubs]
    post_activation_lbs = input_lbs
    post_activation_ubs = input_ubs
    assert len(weights) == num_layers + 1, (len(weights), num_layers)
    for i in range(1, num_layers):
        w = weights[i]
        pre_activation_lbs = torch.where(w > 0, w, 0) @ post_activation_lbs + torch.where(w < 0, w, 0) @ post_activation_ubs + biases[i]
        pre_activation_ubs = torch.where(w > 0, w, 0) @ post_activation_ubs + torch.where(w < 0, w, 0) @ post_activation_lbs + biases[i]
        lbs.append(pre_activation_lbs)
        ubs.append(pre_activation_ubs)
        post_activation_lbs = pre_activation_lbs.clamp(min=0)
        post_activation_ubs = pre_activation_ubs.clamp(min=0)

    return lbs, ubs


class InputBranch:
    input_lbs: List[torch.Tensor]
    input_ubs: List[torch.Tensor]
    params_dict: dict
    resulting_lbs: List[torch.Tensor]
    resulting_ubs: List[torch.Tensor]
    weights: List[torch.Tensor]
    biases: List[torch.Tensor]
    
    def __init__(self, input_lbs, input_ubs, params_dict, resulting_lbs, resulting_ubs, weights, biases) -> None:
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs
        self.params_dict = params_dict
        self.resulting_lbs = resulting_lbs
        self.resulting_ubs = resulting_ubs
        self.weights = weights
        self.biases = biases

    def _create_child(self, x_left: bool, y_left: bool):
        x_input_size = self.input_ubs[0] - self.input_lbs[0]
        y_input_size = self.input_ubs[1] - self.input_lbs[1]
        new_x_lbs = self.input_lbs[0] if x_left else self.input_lbs[0] + x_input_size / 2
        new_x_ubs = self.input_lbs[0] + x_input_size / 2 if x_left else self.input_ubs[0]
        new_y_lbs = self.input_lbs[1] if y_left else self.input_lbs[1] + y_input_size / 2
        new_y_ubs = self.input_lbs[1] + y_input_size / 2 if y_left else self.input_ubs[1]

        new_input_lbs = torch.Tensor([new_x_lbs, new_y_lbs])
        new_input_ubs = torch.Tensor([new_x_ubs, new_y_ubs])

        new_resulting_lbs, new_resulting_ubs = initialize_bounds(len(self.weights) - 1, self.weights, self.biases, new_input_lbs, new_input_ubs)
        new_resulting_lbs = [torch.max(x, y) for x, y in zip(new_resulting_lbs, self.resulting_lbs)]
        new_resulting_ubs = [torch.min(x, y) for x, y in zip(new_resulting_ubs, self.resulting_ubs)]
        new_branch = InputBranch(input_lbs=new_input_lbs, input_ubs=new_input_ubs, params_dict=deepcopy(self.params_dict), resulting_lbs=new_resulting_lbs, resulting_ubs=new_resulting_ubs, weights=self.weights, biases=self.biases)

        return new_branch

    def split(self):
        topleft = self._create_child(True, False)
        topright = self._create_child(False, False)
        bottomleft = self._create_child(True, True)
        bottomright = self._create_child(False, True)

        return [topleft, topright, bottomleft, bottomright]
    


def plot(model: trainer.nn.Sequential, H: torch.Tensor, d: torch.Tensor, approximated_input_bounds: List[ApproximatedInputBound], branch: InputBranch = None):
    plt.rcParams["figure.figsize"] = (8,8)
    plt.cla()

    resolution = 1000
    XX, YY = np.meshgrid(np.linspace(-2, 2, resolution), np.linspace(-2, 2, resolution))
    X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
    y0 = model(X0)
    id = torch.max(y0[:,0], y0[:,1])
    ZZ = (y0[:,2] - id).resize(resolution,resolution).data.numpy()
    output_constraints = torch.all(H.matmul(y0.unsqueeze(-1)).squeeze(-1) + d <= 0, dim=1)
    target_area = (output_constraints).resize(resolution,resolution).data.numpy()
    bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1


    plt.contour(XX,YY,target_area, colors="red", levels=[0,1])
    plt.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))
    plt.axis("equal")

    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.1, 2.1)


    t = np.linspace(0, 2 * math.pi, resolution)
    radius = 0.5
    plt.plot(-1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")
    plt.plot( 1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")

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
                x_vals[0] = new_min_x_val
                # print("1a")
            else:
                assert max_asserted_y_val > max_y_val # b
                # print("1b")
                # plt.plot(np.array([x_vals[0], x_vals[0]]), np.array([max_y_val, max_asserted_y_val]), '--', color="black")
            
            if min_y_val < min_asserted_y_val: # c
                new_max_x_val = (min_asserted_y_val - intercept) / slope
                assert new_max_x_val < x_vals[1]
                # plt.plot(np.array([new_max_x_val, x_vals[1]]), np.array([min_asserted_y_val, min_asserted_y_val]), '--', color="black")
                x_vals[1] = new_max_x_val
                # print("1c", f"{x_vals=}", f"{y_vals=}", f"{asserted_y_vals=}")
            else:
                assert min_asserted_y_val < min_y_val
                # print("1d")
                # plt.plot(np.array([x_vals[1], x_vals[1]]), np.array([min_asserted_y_val, min_y_val]), '--', color="black")
        else:
            assert slope > 0
            if max_y_val > max_asserted_y_val: # a
                new_max_x_val = (max_asserted_y_val - intercept) / slope
                assert new_max_x_val < x_vals[1]
                # plt.plot(np.array([new_max_x_val, x_vals[1]]), np.array([max_asserted_y_val, max_asserted_y_val]), '--', color="black")
                x_vals[1] = new_max_x_val
                # print("2a")
            else:
                assert max_asserted_y_val > max_y_val # b
                # print("2b")
                # plt.plot(np.array([x_vals[1], x_vals[1]]), np.array([max_asserted_y_val, max_y_val]), '--', color="black")
            
            if min_y_val < min_asserted_y_val: # c
                new_min_x_val = (min_asserted_y_val - intercept) / slope
                assert new_min_x_val > x_vals[0]
                # plt.plot(np.array([x_vals[0], new_min_x_val]), np.array([min_asserted_y_val, min_asserted_y_val]), '--', color="black")
                x_vals[0] = new_min_x_val
                # print("2c")
            else:
                assert min_asserted_y_val < min_y_val # d
                # print("2d")
                # plt.plot(np.array([x_vals[0], x_vals[0]]), np.array([min_asserted_y_val, min_y_val]), '--', color="black")
             

        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color="black")

    for approximated_input_bound in approximated_input_bounds:
        c = approximated_input_bound.c
        b = approximated_input_bound.b
        x_vals = np.array([approximated_input_bound.input_lbs[0], approximated_input_bound.input_ubs[0]])
        y_vals = np.array([approximated_input_bound.input_lbs[1], approximated_input_bound.input_ubs[1]])
        from copy import deepcopy
        abline(deepcopy(x_vals), deepcopy(y_vals), -c[0] / c[1], b / c[1])

    if branch is not None:
        plt.plot(np.array([branch.input_lbs[0], branch.input_ubs[0], branch.input_ubs[0], branch.input_lbs[0], branch.input_lbs[0]]),
                 np.array([branch.input_lbs[1], branch.input_lbs[1], branch.input_ubs[1], branch.input_ubs[1], branch.input_lbs[1]]), color="red")

    plt.draw()
    plt.pause(1)