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
            self.model = "toy"
            self.num_epochs = 1
            self.lr = 0.1

    trainer = trainer.Trainer(args())
    trainer.load_model("log/09-16-14:46:07--TEST/weights-last.pt")

    weight1 = trainer.model[1].weight.detach().numpy()
    weight2 = trainer.model[3].weight.detach().numpy()
    weight3 = trainer.model[5].weight.detach().numpy()
    bias1 = trainer.model[1].bias.detach().numpy()
    bias2 = trainer.model[3].bias.detach().numpy()
    bias3 = trainer.model[5].bias.detach().numpy()
    HIDDEN_DIM = 100

    test_label = np.array([0.99, 0.01])

    # Create a new model
    m = gp.Model("verify_input")

    # Create variables
    input = m.addMVar(shape=2, lb=-2, ub=2, name="input")
    x1 = m.addMVar(shape=HIDDEN_DIM, lb=-1e30, ub=1e30, name="x1")
    z1 = m.addMVar(shape=HIDDEN_DIM, lb=-1e30, ub=1e30, name="z1")
    x2 = m.addMVar(shape=HIDDEN_DIM, lb=-1e30, ub=1e30, name="x2")
    z2 = m.addMVar(shape=HIDDEN_DIM, lb=-1e30, ub=1e30, name="z2")
    output = m.addMVar(shape=2, lb=-1e30, ub=1e30, name="output")

    m.setObjective(5 * (input[0] + 1) * (input[0] + 1) + \
                   1 * input[1]       * input[1], GRB.MAXIMIZE)

    m.Params.NonConvex = 2

    m.addConstr(((weight1 @ input) + bias1) == x1)

    for i in range(HIDDEN_DIM):
        m.addConstr(z1[i] == gp.max_(x1[i], constant=0))

    m.addConstr(((weight2 @ z1) + bias2) == x2)

    for i in range(HIDDEN_DIM):
        m.addConstr(z2[i] == gp.max_(x2[i], constant=0))

    m.addConstr(((weight3 @ z2) + bias3) == output)

    m.addConstr(output[0] >= output[1] + 9)
    
    m.optimize()

    for v in m.getVars():

        print('%s %g' % (v.varName, v.x))

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))