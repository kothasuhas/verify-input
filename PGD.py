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
trainer.model.eval()
weight1 = trainer.model[1].weight.detach().numpy()
weight2 = trainer.model[3].weight.detach().numpy()
weight3 = trainer.model[5].weight.detach().numpy()
bias1 = trainer.model[1].bias.detach().numpy()
bias2 = trainer.model[3].bias.detach().numpy()
bias3 = trainer.model[5].bias.detach().numpy()
HIDDEN_DIM1 = 200
HIDDEN_DIM2 = 200

cs = [torch.randn(2) for _ in range(1)]

bs = []

def loss(x, thresh, c):
    y = trainer.model(x)
    return 1000 * torch.nn.functional.relu(thresh - (y[0][0] - y[0][2])) + torch.dot(c, x[0])

for c in cs:
    x = Variable(torch.randn(1, 2), requires_grad=True)
    p = 0.90
    thresh = np.log(p / (1 - p))

    print(thresh)

    best_so_far = 10
    for _ in tqdm(range (10000)):
        l = loss(x, thresh, c)
        l.backward()
        x.data -= 0.0001 * x.grad.data
        x.grad.data.zero_()
        y = trainer.model(x)
        if (y[0][0] - thresh >= y[0][2]):
            best_so_far = min(best_so_far, torch.dot(c, x[0]))
    
    bs.append(best_so_far)

print(cs, bs)
