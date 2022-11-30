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

trainer = trainer.Trainer(args())
trainer.load_model("log/11-02-16:50:02--TEST-3L/weights-last.pt") # 200 200 3
weight0 = trainer.model[1].weight.detach().numpy()
weight1 = trainer.model[3].weight.detach().numpy()
weight2 = trainer.model[5].weight.detach().numpy()
bias0 = trainer.model[1].bias.detach().numpy()
bias1 = trainer.model[3].bias.detach().numpy()
bias2 = trainer.model[5].bias.detach().numpy()
INPUT_DIM   = 2
HIDDEN_DIM1 = 200
HIDDEN_DIM2 = 200
OUTPUT_DIM  = 3

def dotproduct(a, b, len):
    return sum([a[i] * b[i] for i in range(len)])

cs = [torch.randn(2) for _ in range(100)]
# cs = [[0.2326, 1.6094]]

bs = []

for c in tqdm(cs):
    try:
        # Create a new model
        m = gp.Model("verify_input")

        # Set upper and lower bounds
        L = -1000
        U = 1000

        # Create variables
        lambda0 = m.addMVar(shape=HIDDEN_DIM1, lb=-1e30, ub=1e30, name="lambda0")
        lambda1 = m.addMVar(shape=HIDDEN_DIM1, lb=-1e30, ub=0   , name="lambda1")
        lambda2 = m.addMVar(shape=HIDDEN_DIM2, lb=-1e30, ub=1e30, name="lambda2")
        lambda3 = m.addMVar(shape=HIDDEN_DIM2, lb=-1e30, ub=0   , name="lambda3")
        lambda4 = m.addMVar(shape=OUTPUT_DIM , lb=-1e30, ub=1e30, name="lambda4")

        obj_terms = []

        # hidden layer constraints

        for i in range(INPUT_DIM):
            lb = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb)
            m.addConstr(lb <= (c[i].item() - dotproduct(lambda0, weight0[:,i], HIDDEN_DIM1)) * -2)
            m.addConstr(lb <= (c[i].item() - dotproduct(lambda0, weight0[:,i], HIDDEN_DIM1)) * 2)

        for i in range(HIDDEN_DIM1):
            lb1 = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb1)
            m.addConstr(lb1 <= 0)
            m.addConstr(lb1 <= lambda0[i] * L)

            lb2 = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb2)
            m.addConstr(lb2 <= 0)
            m.addConstr(lb2 <= (lambda0[i] - lambda1[i]) * U)

        for i in range(HIDDEN_DIM1):
            lb = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb)
            m.addConstr(lb <= (lambda1[i] - dotproduct(lambda2, weight1[:,i], HIDDEN_DIM2)) * L)
            m.addConstr(lb <= (lambda1[i] - dotproduct(lambda2, weight1[:,i], HIDDEN_DIM2)) * U)

        for i in range(HIDDEN_DIM2):
            lb1 = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb1)
            m.addConstr(lb1 <= 0)
            m.addConstr(lb1 <= lambda2[i] * L)

            lb2 = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb2)
            m.addConstr(lb2 <= 0)
            m.addConstr(lb2 <= (lambda2[i] - lambda3[i]) * U)

        for i in range(HIDDEN_DIM2):
            lb = m.addVar(lb=-1e30, ub=1e30)
            obj_terms.append(lb)
            m.addConstr(lb <= (lambda3[i] - dotproduct(lambda4, weight2[:,i], OUTPUT_DIM)) * L)
            m.addConstr(lb <= (lambda3[i] - dotproduct(lambda4, weight2[:,i], OUTPUT_DIM)) * U)

        # output specification

        p = 0.99
        thresh = np.log(p / (1 - p))

        lb = m.addVar(lb=-1e30, ub=1e30)
        obj_terms.append(lb)
        m.addConstr(lb <= lambda4[0] * U + lambda4[2] * L)
        m.addConstr(lb <= lambda4[0] * U + lambda4[2] * (U - thresh))
        m.addConstr(lb <= lambda4[0] * (L + thresh) + lambda4[2] * L)

        # constant terms

        obj_terms += [-dotproduct(lambda0, bias0, HIDDEN_DIM1),
                      -dotproduct(lambda2, bias1, HIDDEN_DIM2),
                      -dotproduct(lambda4, bias2, OUTPUT_DIM),
                      sum([lambda1[i] * 0.001 for i in range(HIDDEN_DIM1)]),
                      sum([lambda3[i] * 0.001 for i in range(HIDDEN_DIM1)])]

        m.setObjective(sum(obj_terms), GRB.MAXIMIZE)

        m.Params.OutputFlag = 0
        
        m.optimize()

        bs.append(m.getObjective().getValue())

        # for v in m.getVars():
        #     if 'lambda' in v.varName:
        #         print('%s %g' % (v.varName, v.x))

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

grid_dim = 5

XX, YY = np.meshgrid(np.linspace(-grid_dim, grid_dim, 100), np.linspace(-grid_dim, grid_dim, 100))
X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
y0 = trainer.model(X0)
id = torch.max(y0[:,0], y0[:,1])
ZZ = (y0[:,2] - id).resize(100,100).data.numpy()
bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1

fig, ax = plt.subplots(figsize=(8,8))
ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))
ax.axis("equal")

plt.xlim(-grid_dim, grid_dim)
plt.ylim(-grid_dim, grid_dim)

t = np.linspace(0, 2 * math.pi, 100)
radius = 0.5
plt.plot(-1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")
plt.plot( 1 + radius * np.cos(t), 0 + radius * np.sin(t), color="blue")

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color="black")

for c, b in list(zip(cs, bs)):
    abline(-c[0] / c[1], b / c[1])

print(cs, bs)

plt.show()