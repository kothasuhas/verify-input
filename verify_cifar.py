import os, shutil, json, datetime, time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch

import core.trainer as trainer
import core.data as data

from tqdm import tqdm as tqdm

try:
    NUM_POINTS = 2
    # # Load data and model
    class args():
        def __init__(self):
            self.model = "mlp"
            self.num_epochs = 1
            self.lr = 0.1
            pass

    trainer = trainer.Trainer(args())
    trainer.load_model("log/07-03-22:41:42--TEST/weights-best.pt")

    weight1 = trainer.model[1].weight.detach().numpy()
    weight2 = trainer.model[3].weight.detach().numpy()
    weight3 = trainer.model[5].weight.detach().numpy()
    bias1 = trainer.model[1].bias.detach().numpy()
    bias2 = trainer.model[3].bias.detach().numpy()
    bias3 = trainer.model[5].bias.detach().numpy()

    train_dataset, test_dataset, train_dataloader, test_dataloader = data.load_data(
      os.path.join('cifar10_data', 'cifar10s'), NUM_POINTS, use_augmentation=False
    )

    train_data_numpy = next(iter(train_dataloader))[0].reshape(NUM_POINTS, -1).numpy()

    # Create a new model
    m = gp.Model("verify_input")
    # m.Params.DualReductions = 0

    # Create variables
    x0 = m.addMVar(shape=3072, lb=0, ub=1, name="x0")
    x1 = m.addMVar(shape=256 , lb=-1e30, ub=1e30, name="x1")
    x2 = m.addMVar(shape=256 , lb=-1e30, ub=1e30, name="x2")
    x3 = m.addMVar(shape=128 , lb=-1e30, ub=1e30, name="x3")
    x4 = m.addMVar(shape=128 , lb=-1e30, ub=1e30, name="x4")
    x5 = m.addMVar(shape=10  , lb=-1e30, ub=1e30, name="x5")

    delta = m.addVar(name="delta")
    dist = m.addMVar(shape=(NUM_POINTS,3072), vtype=GRB.CONTINUOUS, lb=-1e30, ub=1e30, name="dist")
    abs_dist = m.addMVar(shape=(NUM_POINTS,3072), name="abs_dist")

    m.setObjective(delta, GRB.MAXIMIZE)

    # 1) Adversarial input is far away from training points
    for i in range(3072):
        for a in range(NUM_POINTS):
            m.addConstr((dist[a][i] + x0[i]) == (train_data_numpy[a][i]))
            m.addGenConstrAbs(abs_dist[a][i], dist[a][i])
            m.addConstr(delta <= abs_dist[a][i])

    # 2) Input maps to the correct output
    m.addConstr(((weight1 @ x0) + bias1) == x1)

    for i in range(256):
        m.addConstr(x2[i] == gp.max_(x1[i], constant=0))

    m.addConstr(((weight2 @ x2) + bias2) == x3)

    for i in range(128):
        m.addConstr(x4[i] == gp.max_(x3[i], constant=0))

    m.addConstr(((weight3 @ x4) + bias3) == x5)

    for i in range(10):
        m.addConstr(x5[i] == 0.5)
    
    m.optimize()
    # print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))