import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

import core.trainer as trainer

from tqdm import tqdm as tqdm

class args():
    def __init__(self):
        self.model = "toy"
        self.num_epochs = 1
        self.lr = 0.1

trainer = trainer.Trainer(args())
trainer.load_model("log/11-02-16:50:02--TEST-3L/weights-last.pt") # 200 200 3
trainer.model.eval()

def initialize(model, h, L):
    weights = [None]
    biases = [None]
    lbs = [torch.full((2,), -2.0)]
    ubs = [torch.full((2,), 2.0)]
    alphas = [None]
    
    for i in range(1, L+1):
        weights.append(model[2*i - 1].weight.detach())
        biases.append(model[2*i - 1].bias.detach())

    for i in range(1, L):
        alphas.append(torch.full((weights[i].size(0),), 0.5))
        lbs.append(torch.full((weights[i].size(0),), -4.0))
        ubs.append(torch.full((weights[i].size(0),), 4.0))

    gamma = torch.full((1,), 0.01)

    gamma.requires_grad = True
    alphas[1].requires_grad = True
    alphas[2].requires_grad = True

    weights[L] = torch.matmul(h.transpose(0, 1), weights[L])
    biases[L]  = torch.matmul(h.transpose(0, 1), biases[L])

    return gamma, alphas, weights, biases, lbs, ubs

def get_diagonals(weights, lbs, ubs, gamma, alphas, L):
    A = [None for _ in range(L)]
    D = [None for _ in range(L)]

    for i in range(L-1, 0, -1):
        if i == L-1:
            A[i] = gamma * weights[L]
        else:
            A[i] = A[i+1].matmul(D[i+1]).matmul(weights[i+1])

        D[i] = torch.zeros(weights[i].size(0), weights[i].size(0))

        for j in range(D[i].size(0)):
            if lbs[i][j] >= 0:   # ReLU always on
                D[i][j][j] = 1
            elif ubs[i][j] <= 0: # ReLU always off
                D[i][j][j] = 0
            elif A[i][0][j] >= 0:   # use ReLU lower bound
                D[i][j][j] = alphas[i][j]
            elif A[i][0][j] < 0:    # use ReLU upper bound
                D[i][j][j] = ubs[i][j] / (ubs[i][j] - lbs[i][j])

    return A, D

def get_bias_lbs(A, lbs, ubs, L):
    bias_lbs = [None]

    for i in range(1, L):
        bias_lbs.append(torch.zeros(A[i].size(1)))
        for j in range(bias_lbs[i].size(0)):
            if lbs[i][j] >= 0 or ubs[i][j] <= 0: # Stable neuron
                bias_lbs[i][j] = 0
            elif A[i][0][j] >= 0:
                bias_lbs[i][j] = 0
            elif A[i][0][j] < 0:
                bias_lbs[i][j] = - (ubs[i][j] * lbs[i][j]) / (ubs[i][j] - lbs[i][j])

    return bias_lbs

def init_Omega(weights, biases, gamma, D, L):
    def Omega(end, start):
        assert end >= start
        if end == start: return torch.eye(biases[start].size(0))
        return (gamma if end == L else 1) * weights[end].matmul(D[end - 1]).matmul(Omega(end - 1, start))
    return Omega

L = 3

h = torch.zeros(3, 1)
h[0][0] = -1
h[2][0] = 1

gamma, alphas, weights, biases, lbs, ubs = initialize(trainer.model, h, L)

def get_obj(gamma, alphas, lbs, ubs, c, layeri=0):
    A, D = get_diagonals(weights, lbs, ubs, gamma, alphas, L)
    bias_lbs = get_bias_lbs(A, lbs, ubs, L)

    def biases_scaled(i):
        return (biases[i] if i < L else gamma * biases[L])

    def weights_scaled(i):
        return (weights[i] if i < L else gamma * weights[L])

    Omega = init_Omega(weights, biases, gamma, D, L)

    a_crown = Omega(L, layeri + 1).matmul(weights[layeri + 1])
    c_crown = sum([Omega(L, i).matmul(biases_scaled(i)) for i in range(layeri + 1, L + 1)]) \
            + sum([Omega(L, i).matmul(weights_scaled(i)).matmul(bias_lbs[i - 1]) for i in range(layeri + 2, L + 1)])

    p = 0.9
    thresh = np.log(p / (1 - p))
    x_0 = (ubs[layeri] + lbs[layeri]) / 2.0
    eps = (ubs[layeri] - lbs[layeri]) / 2.0

    return -torch.abs(c + a_crown).squeeze(0).dot(eps) + (c + a_crown).matmul(x_0) + c_crown + gamma * thresh 

optim = torch.optim.SGD([gamma, alphas[1], alphas[2]], lr=0.01, momentum=0.9)
layeri = 1

for _ in range(20):
    optim.zero_grad()
    c = torch.Tensor([-0.2326, -1.6094]) if layeri == 0 else torch.Tensor([1 if i == 0 else 0 for i in range(200)])
    loss = -get_obj(gamma, alphas, lbs, ubs, c, layeri=layeri)
    print(-loss.item(), gamma.item(), alphas[2][0].item())
    loss.backward()
    optim.step()

    with torch.no_grad():
        gamma.data = torch.clamp(gamma.data, min=0)
        alphas[1].data = alphas[1].data.clamp(min=0.0, max=1.0)
        alphas[2].data = alphas[2].data.clamp(min=0.0, max=1.0)
