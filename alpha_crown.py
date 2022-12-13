import gurobipy as gp
from gurobipy import GRB
import numpy as np
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
    alphas = [None]
    
    for i in range(1, L+1):
        weights.append(model[2*i - 1].weight.detach())
        biases.append(model[2*i - 1].bias.detach())

    weights[L] = torch.matmul(h.transpose(0, 1), weights[L])
    biases[L]  = torch.matmul(h.transpose(0, 1), biases[L])

    for i in range(1, L):
        alphas.append(torch.full((weights[i].size(0),), 0.5))

    gamma = torch.full((1,), 0.01)

    gamma.requires_grad = True
    alphas[1].requires_grad = True
    alphas[2].requires_grad = True

    return gamma, alphas, weights, biases

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

def get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L):
    A, D = get_diagonals(weights, lbs, ubs, gamma, alphas, L)
    bias_lbs = get_bias_lbs(A, lbs, ubs, L)

    def biases_scaled(i):  return (biases[i]  if i < L else gamma * biases[L])
    def weights_scaled(i): return (weights[i] if i < L else gamma * weights[L])

    Omega = init_Omega(weights, biases, gamma, D, L)

    a_crown = Omega(L, 1).matmul(weights[1])
    c_crown = sum([Omega(L, i).matmul(biases_scaled(i)) for i in range(1, L + 1)]) \
            + sum([Omega(L, i).matmul(weights_scaled(i)).matmul(bias_lbs[i - 1]) for i in range(2, L + 1)])

    return a_crown, c_crown

def get_obj_all(weights, biases, gamma, alphas, lbs, ubs, c, thresh, L):
    a_crown, c_crown = get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L)
    a_crown += c

    x_0 = (ubs[0] + lbs[0]) / 2.0
    eps = (ubs[0] - lbs[0]) / 2.0

    return -torch.abs(a_crown).squeeze(0).dot(eps) + a_crown.matmul(x_0) + c_crown + gamma * thresh 

def optimize_bound(weights, biases, gamma, alphas, lbs, ubs, thresh, L, layeri, neuron, direction):
    a_crown_full, c_crown_full = get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L)

    L1 = layeri
    weights1 = weights[:layeri+1]
    biases1  = biases[:layeri+1]
    ubs1 = ubs[:layeri]
    lbs1 = lbs[:layeri]
    alphas1 = alphas[:layeri]

    c = torch.zeros(weights1[-1].size(0))
    c[neuron] = (1 if direction == "lbs" else -1)
    weights1[-1] = c.unsqueeze(0).matmul(weights1[-1])
    biases1[-1]  = c.unsqueeze(0).matmul(biases1[-1])
    
    a_crown_partial, c_crown_partial = get_crown_bounds(weights1, biases1, 1.0, alphas1, lbs1, ubs1, L1)
    
    a_crown = a_crown_partial + a_crown_full
    c_crown = c_crown_partial + c_crown_full

    x_0 = (ubs[0] + lbs[0]) / 2.0
    eps = (ubs[0] - lbs[0]) / 2.0

    return -torch.abs(a_crown).squeeze(0).dot(eps) + a_crown.matmul(x_0) + c_crown + gamma * thresh 

gamma, alphas, weights, biases = initialize(trainer.model, torch.Tensor([[-1], [0], [1]]), 3)
lbs = [torch.full((2,), -2.0)]
ubs = [torch.full((2,), 2.0)]
for i in range(1, 3):
    lbs.append(torch.full((weights[i].size(0),), -4.0))
    ubs.append(torch.full((weights[i].size(0),), 4.0))

p = 0.9
thresh = np.log(p / (1 - p))
bounds = {"lbs" : lbs, "ubs" : ubs}

for _ in range(10):
    for direction, layeri in [("lbs", 2), ("ubs", 2), ("lbs", 1), ("ubs", 1)]:
        for neuron in tqdm(range(200)):
            gamma, alphas, weights, biases = initialize(trainer.model, torch.Tensor([[-1], [0], [1]]), 3)
            optim = torch.optim.SGD([gamma, alphas[1], alphas[2]], lr=0.1, momentum=0.9, maximize=True)
            if bounds == "lbs" and (bounds[direction][layeri][neuron] >= 0.0): continue
            if bounds == "ubs" and (bounds[direction][layeri][neuron] <= 0.0): continue
            for _ in range(10):
                optim.zero_grad()
                loss = optimize_bound(weights, biases, gamma, alphas, lbs, ubs, thresh, 3, layeri, neuron, direction)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    if direction == "lbs":
                        bounds[direction][layeri][neuron] = torch.max(bounds[direction][layeri][neuron], loss.detach())
                    else:
                        bounds[direction][layeri][neuron] = torch.min(bounds[direction][layeri][neuron], -loss.detach())
                    gamma.data = torch.clamp(gamma.data, min=0)
                    alphas[1].data = alphas[1].data.clamp(min=0.0, max=1.0)
                    alphas[2].data = alphas[2].data.clamp(min=0.0, max=1.0)

    # Create a new model
    try:
        m = gp.Model("verify_input")

        # Create variables
        input = m.addMVar(shape=2, lb=-2, ub=2, name="input")
        x0 = m.addMVar(shape=200, lb=-1e30, ub=1e30, name="x0")
        z1 = m.addMVar(shape=200, lb=-1e30, ub=1e30, name="z1")
        x1 = m.addMVar(shape=200, lb=-1e30, ub=1e30, name="x1")
        z2 = m.addMVar(shape=200, lb=-1e30, ub=1e30, name="z2")
        output = m.addVar(lb=-1e30, ub=1e30, name="output")

        weight1 = weights[1].numpy()
        weight2 = weights[2].numpy()
        weight3 = weights[3].numpy()
        bias1 = biases[1].numpy()
        bias2 = biases[2].numpy()
        bias3 = biases[3].numpy()
        lbs1 = lbs[1].numpy()
        lbs2 = lbs[2].numpy()
        ubs1 = ubs[1].numpy()
        ubs2 = ubs[2].numpy()

        c = [-0.2326, -1.6094]
        m.setObjective(c[0] * input[0] + c[1] * input[1], GRB.MINIMIZE)

        m.Params.OutputFlag = 0

        m.addConstr(((weight1 @ input) + bias1) == x0)

        for i in range(200):
            assert bounds["lbs"][1][i] <= bounds["ubs"][1][i]
            if bounds["ubs"][1][i] <= 0:
                m.addConstr(z1[i] == 0)
            elif bounds["lbs"][1][i] >= 0:
                m.addConstr(z1[i] == x0[i])
            else:
                m.addConstr(z1[i] >= 0)
                m.addConstr(z1[i] >= x0[i])
                m.addConstr(z1[i] <= x0[i] * ubs1[i] / (ubs1[i] - lbs1[i]) - (lbs1[i] * ubs1[i]) / (ubs1[i] - lbs1[i]))

        m.addConstr(((weight2 @ z1) + bias2) == x1)

        for i in range(200):
            assert bounds["lbs"][2][i] <= bounds["ubs"][2][i]
            if bounds["ubs"][2][i] <= 0:
                m.addConstr(z2[i] == 0)
            elif bounds["lbs"][2][i] >= 0:
                m.addConstr(z2[i] == x1[i])
            else:
                m.addConstr(z2[i] >= 0)
                m.addConstr(z2[i] >= x1[i])
                m.addConstr(z2[i] <= x1[i] * ubs2[i] / (ubs2[i] - lbs2[i]) - (lbs2[i] * ubs2[i]) / (ubs2[i] - lbs2[i]))

        m.addConstr(((weight3 @ z2) + bias3) == output)
        m.addConstr(output + thresh <= 0)
        m.optimize()
        print(m.getObjective().getValue())

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))