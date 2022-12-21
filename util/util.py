from typing import List

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import torch
from torch.autograd import Variable

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
    return model[layer*2+1].weight.detach().numpy().shape[0]

def get_optimal_grb_model(model: trainer.nn.Sequential, h: torch.Tensor, thresh: float):
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
    m.addConstr(h.detach().numpy().T @ xs[-1] + thresh <= 0)

    m.Params.NonConvex = 2

    return m, xs, zs

def get_triangle_grb_model(model: trainer.nn.Sequential, ubs, lbs, h: torch.Tensor, thresh: float):
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
            assert l <= u
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
    m.addConstr(h.detach().numpy().T @ xs[-1] + thresh <= 0)

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

def plot(model: trainer.nn.Sequential, thresh: float, cs: List[torch.Tensor], bss: List[List[float]]):
    resolution = 1000
    XX, YY = np.meshgrid(np.linspace(-2, 2, resolution), np.linspace(-2, 2, resolution))
    X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
    y0 = model(X0)
    id = torch.max(y0[:,0], y0[:,1])
    ZZ = (y0[:,2] - id).resize(resolution,resolution).data.numpy()
    target_area = (y0[:,0] >= y0[:,2] + thresh).resize(resolution,resolution).data.numpy()
    bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1

    fig, ax = plt.subplots(figsize=(8,8))
    ax.contour(XX,YY,target_area, colors="red", levels=[0,1])
    ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))
    ax.axis("equal")

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    t = np.linspace(0, 2 * math.pi, resolution)
    radius = 0.5
    plt.plot(-1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")
    plt.plot( 1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")

    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color="black")

    for c, bs in list(zip(cs, bss)):
        for b in bs:
            abline(-c[0] / c[1], b / c[1])

    print(cs, bss)

    plt.show()