from typing import List, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch

import core.trainer as trainer
from .model_utils import get_num_layers

def _get_grb_model(model: trainer.nn.Sequential, layers: int, input_lbs: List[float], input_ubs: List[float]):
    m = gp.Model("verify_input")
    m.Params.OutputFlag = 0

    # Create variables
    input = m.addMVar(shape=2, lb=input_lbs, ub=input_ubs, name="input")
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

def get_optimal_grb_model(model: trainer.nn.Sequential, H: torch.Tensor, d: torch.Tensor, input_lbs: List[float], input_ubs: List[float]):
    layers = get_num_layers(model)

    m, xs, zs = _get_grb_model(model, layers, input_lbs, input_ubs)
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
    m.addConstr(H.detach().numpy() @ xs[-1] + d.detach().numpy() <= 0)

    return m, xs, zs

def get_triangle_grb_model(model: trainer.nn.Sequential, ubs, lbs, H: torch.Tensor, d: torch.Tensor, input_lbs: List[float], input_ubs: List[float]):
    layers = get_num_layers(model)
        
    m, xs, zs = _get_grb_model(model, layers, input_lbs, input_ubs)

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

def get_optimized_grb_result(m: gp.Model, c, inputs) -> Optional[float]:
    m.setObjective(c[0] * inputs[0] + c[1] * inputs[1], GRB.MINIMIZE)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        return m.getObjective().getValue()
    elif m.status == GRB.INFEASIBLE:
        return None
    else:
        sc = gp.StatusConstClass
        d = {sc.__dict__[k]: k for k in sc.__dict__.keys() if k[0] >= 'A' and k[0] <= 'Z'}
        print(f"Gurobi returned a non optimal result. Error description: {d[m.status]}. Aborting.")
        exit()
