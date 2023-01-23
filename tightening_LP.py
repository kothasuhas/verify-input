from tkinter import HIDDEN

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import math
import torch
from torch.autograd import Variable
from tqdm import tqdm

import core.trainer as trainer
import core.data as data

import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm

class args():
    def __init__(self):
        self.model = "toy"
        self.num_epochs = 1
        self.lr = 0.1
        self.optimizer = "Adam"
        self.sched_pct = 0.0

trainer = trainer.Trainer(args())
trainer.load_model("test-weights.pt") # 200 200 3
weight1 = trainer.model[1].weight.detach().numpy()
weight2 = trainer.model[3].weight.detach().numpy()
weight3 = trainer.model[5].weight.detach().numpy()
bias1 = trainer.model[1].bias.detach().numpy()
bias2 = trainer.model[3].bias.detach().numpy()
bias3 = trainer.model[5].bias.detach().numpy()
HIDDEN_DIM1 = 200
HIDDEN_DIM2 = 200

# cs = [torch.randn(2) for _ in range(20)]
cs = [[-0.2326, -1.6094]]

bs = []

bounds = {
    "ubs1": [4 for _ in range(200)],
    "lbs1": [-4 for _ in range(200)],
    "ubs2": [4 for _ in range(200)],
    "lbs2": [-4 for _ in range(200)],
}

for k in tqdm(range(10)):
    for bound, layer in [("ubs2", "x1"), ("lbs2", "x1"), ("ubs1", "x0"), ("lbs1", "x0")]:
        for k in range(200):
            for c in cs:
                try:
                    # Create a new model
                    m = gp.Model("verify_input")

                    # Create variables
                    input = m.addMVar(shape=2, lb=-2, ub=2, name="input")
                    x0 = m.addMVar(shape=HIDDEN_DIM1, lb=-1e30, ub=1e30, name="x0")
                    z1 = m.addMVar(shape=HIDDEN_DIM1, lb=-1e30, ub=1e30, name="z1")
                    x1 = m.addMVar(shape=HIDDEN_DIM2, lb=-1e30, ub=1e30, name="x1")
                    z2 = m.addMVar(shape=HIDDEN_DIM2, lb=-1e30, ub=1e30, name="z2")
                    output = m.addMVar(shape=3, lb=-1e30, ub=1e30, name="output")

                    vars = {
                        "x0" : x0,
                        "x1" : x1
                    }

                    # m.setObjective(c[0] * input[0] + c[1] * input[1], GRB.MINIMIZE)
                    if "ubs" in bound:
                        m.setObjective(vars[layer][k], GRB.MAXIMIZE)
                    else:
                        m.setObjective(vars[layer][k], GRB.MINIMIZE)

                    m.Params.OutputFlag = 0
                    # m.Params.NonConvex = 2

                    m.addConstr(((weight1 @ input) + bias1) == x0)

                    for i in range(HIDDEN_DIM2):
                        assert bounds["lbs1"][i] <= bounds["ubs1"][i]
                        # m.addConstr(z1[i] == gp.max_(x0[i], constant=0))
                        if bounds["ubs1"][i] <= 0:
                            m.addConstr(z1[i] == 0)
                        elif bounds["lbs1"][i] >= 0:
                            m.addConstr(z1[i] == x0[i])
                        else:
                            m.addConstr(z1[i] >= 0)
                            m.addConstr(z1[i] >= x0[i])
                            m.addConstr(z1[i] <= x0[i] * bounds["ubs1"][i] / (bounds["ubs1"][i] - bounds["lbs1"][i]) - (bounds["lbs1"][i] * bounds["ubs1"][i]) / (bounds["ubs1"][i] - bounds["lbs1"][i]))

                    m.addConstr(((weight2 @ z1) + bias2) == x1)

                    for i in range(HIDDEN_DIM2):
                        assert bounds["lbs2"][i] <= bounds["ubs2"][i]
                        # m.addConstr(z2[i] == gp.max_(x1[i], constant=0))
                        if bounds["ubs2"][i] <= 0:
                            m.addConstr(z2[i] == 0)
                        elif bounds["lbs2"][i] >= 0:
                            m.addConstr(z2[i] == x1[i])
                        else:
                            m.addConstr(z2[i] >= 0)
                            m.addConstr(z2[i] >= x1[i])
                            m.addConstr(z2[i] <= x1[i] * bounds["ubs2"][i] / (bounds["ubs2"][i] - bounds["lbs2"][i]) - (bounds["lbs2"][i] * bounds["ubs2"][i]) / (bounds["ubs2"][i] - bounds["lbs2"][i]))

                    m.addConstr(((weight3 @ z2) + bias3) == output)

                    p = 0.90
                    thresh = np.log(p / (1 - p))

                    m.addConstr(output[0] >= output[2] + thresh)
                    
                    m.optimize()

                    bounds[bound][k] = m.getObjective().getValue()

                except gp.GurobiError as e:
                    print('Error code ' + str(e.errno) + ": " + str(e))

    for c in cs:
        try:
            # Create a new model
            m = gp.Model("verify_input")

            # Create variables
            input = m.addMVar(shape=2, lb=-2, ub=2, name="input")
            x0 = m.addMVar(shape=HIDDEN_DIM1, lb=-1e30, ub=1e30, name="x0")
            z1 = m.addMVar(shape=HIDDEN_DIM1, lb=-1e30, ub=1e30, name="z1")
            x1 = m.addMVar(shape=HIDDEN_DIM2, lb=-1e30, ub=1e30, name="x1")
            z2 = m.addMVar(shape=HIDDEN_DIM2, lb=-1e30, ub=1e30, name="z2")
            output = m.addMVar(shape=3, lb=-1e30, ub=1e30, name="output")

            m.setObjective(c[0] * input[0] + c[1] * input[1], GRB.MINIMIZE)

            m.Params.OutputFlag = 0
            # m.Params.NonConvex = 2

            m.addConstr(((weight1 @ input) + bias1) == x0)

            for i in range(HIDDEN_DIM2):
                assert bounds["lbs1"][i] <= bounds["ubs1"][i]
                # m.addConstr(z1[i] == gp.max_(x0[i], constant=0))
                if bounds["ubs1"][i] <= 0:
                    m.addConstr(z1[i] == 0)
                elif bounds["lbs1"][i] >= 0:
                    m.addConstr(z1[i] == x0[i])
                else:
                    m.addConstr(z1[i] >= 0)
                    m.addConstr(z1[i] >= x0[i])
                    m.addConstr(z1[i] <= x0[i] * bounds["ubs1"][i] / (bounds["ubs1"][i] - bounds["lbs1"][i]) - (bounds["lbs1"][i] * bounds["ubs1"][i]) / (bounds["ubs1"][i] - bounds["lbs1"][i]))

            m.addConstr(((weight2 @ z1) + bias2) == x1)

            for i in range(HIDDEN_DIM2):
                assert bounds["lbs2"][i] <= bounds["ubs2"][i]
                # m.addConstr(z2[i] == gp.max_(x1[i], constant=0))
                if bounds["ubs2"][i] <= 0:
                    m.addConstr(z2[i] == 0)
                elif bounds["lbs2"][i] >= 0:
                    m.addConstr(z2[i] == x1[i])
                else:
                    m.addConstr(z2[i] >= 0)
                    m.addConstr(z2[i] >= x1[i])
                    m.addConstr(z2[i] <= x1[i] * bounds["ubs2"][i] / (bounds["ubs2"][i] - bounds["lbs2"][i]) - (bounds["lbs2"][i] * bounds["ubs2"][i]) / (bounds["ubs2"][i] - bounds["lbs2"][i]))

            m.addConstr(((weight3 @ z2) + bias3) == output)

            p = 0.90
            thresh = np.log(p / (1 - p))

            m.addConstr(output[0] >= output[2] + thresh)
            
            m.optimize()
            print(m.getObjective().getValue())
            bs.append(m.getObjective().getValue())

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            
print(bs)