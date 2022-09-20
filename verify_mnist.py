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
    NUM_POINTS = 20
    # Load data and model
    class args():
        def __init__(self):
            self.model = "mlp"
            self.num_epochs = 1
            self.lr = 0.1
            pass

    trainer = trainer.Trainer(args())
    trainer.load_model("log/weights-best.pt")

    weight1 = trainer.model[1].weight.detach().numpy()
    weight2 = trainer.model[3].weight.detach().numpy()
    bias1 = trainer.model[1].bias.detach().numpy()
    bias2 = trainer.model[3].bias.detach().numpy()

    train_dataset, test_dataset, train_dataloader, test_dataloader = data.load_data(
      os.path.join('mnist_data'), NUM_POINTS, use_augmentation=False
    )

    train_data_numpy = next(iter(train_dataloader))[0].reshape(NUM_POINTS, -1).numpy()

    test_label = np.array([-1.0067548, -3.894588, 1.9623046 , 8.370875  , -15.300852, 
                           16.84495  , -7.976923, 0.82987297, -2.6402698, 3.6777704  ])

    print(weight2 @ (np.maximum((weight1 @ train_data_numpy[0] + bias1), 0)) + bias2)

    # Create a new model
    m = gp.Model("verify_input")

    # Create variables
    x0 = m.addMVar(shape=784, lb=0    , ub=1   , name="x0")
    x1 = m.addMVar(shape=20 , lb=-1e30, ub=1e30, name="x1")
    x2 = m.addMVar(shape=20 , lb=-1e30, ub=1e30, name="x2")
    x3 = m.addMVar(shape=10 , lb=-1e30, ub=1e30, name="x3")

    delta = m.addVar(name="delta")
    dist = m.addMVar(shape=(NUM_POINTS,784), vtype=GRB.CONTINUOUS, lb=-1e30, ub=1e30, name="dist")
    abs_dist = m.addMVar(shape=(NUM_POINTS,784), name="abs_dist")
    linf_dist = m.addMVar(shape=(NUM_POINTS), name="linf_dist")

    m.setObjective(delta, GRB.MAXIMIZE)

    # 1) Adversarial input is far away from training points
    for a in range(NUM_POINTS):
        for i in range(784):
            m.addConstr((dist[a][i] + x0[i]) == (train_data_numpy[a][i]))
            m.addGenConstrAbs(abs_dist[a][i], dist[a][i])
        m.addConstr(linf_dist[a] == gp.max_([abs_dist[a][i] for i in range(784)]))
        m.addConstr(delta <= linf_dist[a])

    # 2) Input maps to the correct output
    m.addConstr(((weight1 @ x0) + bias1) == x1)

    for i in range(20):
        m.addConstr(x2[i] == gp.max_(x1[i], constant=0))

    m.addConstr(((weight2 @ x2) + bias2) == x3)

    m.addConstr(x3 == test_label)
    
    m.optimize()

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))