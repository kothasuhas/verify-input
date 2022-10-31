import os, shutil, json, datetime, time
from tkinter import HIDDEN

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch

import core.trainer as trainer
import core.data as data

from tqdm import tqdm as tqdm

try:
    class args():
        def __init__(self):
            self.model = "vae"
            self.num_epochs = 1
            self.lr = 0.1

    trainer = trainer.Trainer(args())

    trainer.load_model("log/vae_weights.pt")

    weight1 = trainer.model.encoder.fc1.weight.detach().numpy()
    weight2 = trainer.model.encoder.fc2.weight.detach().numpy()
    weight_mu = trainer.model.encoder.fc_mu.weight.detach().numpy()
    bias1 = trainer.model.encoder.fc1.bias.detach().numpy()
    bias2 = trainer.model.encoder.fc2.bias.detach().numpy()
    bias_mu = trainer.model.encoder.fc_mu.bias.detach().numpy()

    # Create a new model
    m = gp.Model("verify_input")

    # Create variables
    input = m.addMVar(shape=256, lb=-4, ub=4, name="input")
    x1 = m.addMVar(shape=128, lb=-1e30, ub=1e30, name="x1")
    z1 = m.addMVar(shape=128, lb=-1e30, ub=1e30, name="z1")
    x2 = m.addMVar(shape=4, lb=-1e30, ub=1e30, name="x2")
    z2 = m.addMVar(shape=4, lb=-1e30, ub=1e30, name="z2")
    output = m.addMVar(shape=2, lb=-1e30, ub=1e30, name="output")

    v = m.addVar(name="v")
    m.addConstr(v <= 5 * (input[0] + 1) * (input[0] + 1))
    m.addConstr(v <= 5 * (input[0] - 1) * (input[0] - 1))

    # TODO: fix
    m.setObjective(v \
                # + sum([input[i] * input[i] for i in range(1, 256)]) \
                    , GRB.MAXIMIZE)

    m.Params.NonConvex = 2

    # feed forward constraints
    m.addConstr(((weight1 @ input) + bias1) == x1)
    for i in range(128): m.addConstr(z1[i] == gp.max_(x1[i], constant=0))
    m.addConstr(((weight2 @ z1) + bias2) == x2)
    for i in range(4): m.addConstr(z2[i] == gp.max_(x2[i], constant=0))
    m.addConstr(((weight_mu @ z2) + bias_mu) == output)

    m.addConstr(output[0] * output[0] + output[1] * output[1] <= 0.01)
    
    m.optimize()

    for v in m.getVars():
        if 'input' in v.varName or 'output' in v.varName:
            print('%s %g' % (v.varName, v.x))

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))