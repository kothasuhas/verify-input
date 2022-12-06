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

for c in tqdm(cs):
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

        m.setObjective(c[0] * input[0] + c[1] * input[1], GRB.MAXIMIZE)

        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2

        m.addConstr(((weight1 @ input) + bias1) == x0)

        for i in range(HIDDEN_DIM1):
            m.addConstr(z1[i] == gp.max_(x0[i], constant=0))

        m.addConstr(((weight2 @ z1) + bias2) == x1)

        for i in range(HIDDEN_DIM2):
            m.addConstr(z2[i] == gp.max_(x1[i], constant=0))

        m.addConstr(((weight3 @ z2) + bias3) == output)

        p = 0.90
        thresh = np.log(p / (1 - p))

        # m.addConstr(output[0] >= output[1])
        # m.addConstr(output[0] >= output[1] + np.log(p / (1 - p)))
        m.addConstr(output[0] >= output[2] + thresh)
        # m.addConstr(output[0] <= output[1] + np.log(p / (1 - p)))
        
        m.optimize()

        bs.append(m.getObjective().getValue())

        # for v in m.getVars():
        #     if 'input' in v.varName or 'output' in v.varName:
        #         print('%s %g' % (v.varName, v.x))

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

XX, YY = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
y0 = trainer.model(X0)
id = torch.max(y0[:,0], y0[:,1])
ZZ = (y0[:,2] - id).resize(100,100).data.numpy()
bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1

fig, ax = plt.subplots(figsize=(8,8))
ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))
ax.axis("equal")

plt.xlim(-2, 2)
plt.ylim(-2, 2)

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