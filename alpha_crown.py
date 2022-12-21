import itertools

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from tqdm import tqdm

import core.trainer as trainer


from util.util import get_num_layers, get_num_neurons, plot, get_optimized_grb_result, get_triangle_grb_model

class args():
    def __init__(self):
        self.model = "toy"
        self.num_epochs = 1
        self.lr = 0.1

trainer = trainer.Trainer(args())
trainer.load_model("test-weights.pt") # 200 200 3
trainer.model.eval()

def initialize_weights(model, h, L):
    weights = [None]
    biases = [None]
    
    for i in range(1, L+1):
        weights.append(model[2*i - 1].weight.detach())
        biases.append(model[2*i - 1].bias.detach())

    weights[L] = torch.matmul(h.transpose(0, 1), weights[L])
    biases[L]  = torch.matmul(h.transpose(0, 1), biases[L])

    return weights, biases

def initialize_params(L, weights):
    alphas = [None]

    for i in range(1, L):
        alphas.append(torch.full((weights[i].size(0),), 0.5))
        alphas[-1].requires_grad = True

    gamma = torch.full((1,), 0.01)
    gamma.requires_grad = True

    return gamma, alphas

def get_diagonals(weights, lbs, ubs, alphas, L):
    A = [None for _ in range(L)]
    D = [None for _ in range(L)]
    assert len(weights) == L + 1
    for i in range(L-1, 0, -1):  # 1, ..., L-1  -> entry L not used
        if i == L-1:
            A[i] = weights[L]
        else:
            A[i] = A[i+1].matmul(D[i+1]).matmul(weights[i+1])

        D[i] = torch.zeros(weights[i].size(0), weights[i].size(0))

        for j in range(D[i].size(0)):
            if lbs[i][j] >= 0:   # ReLU always on
                diagonal_entry = 1
            elif ubs[i][j] <= 0: # ReLU always off
                diagonal_entry = 0
            elif A[i][0][j] >= 0:   # use ReLU lower bound
                diagonal_entry = alphas[i][j]
            else:    # use ReLU upper bound
                assert A[i][0][j] < 0
                diagonal_entry = ubs[i][j] / (ubs[i][j] - lbs[i][j])
            D[i][j][j] = diagonal_entry

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

def init_Omega(weights, biases, D, L):
    def Omega(end, start):
        assert end >= start
        if end == start: return torch.eye(biases[start].size(0))
        return weights[end].matmul(D[end - 1]).matmul(Omega(end - 1, start))
    return Omega

def get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L):
    A, D = get_diagonals(weights, lbs, ubs, alphas, L)
    bias_lbs = get_bias_lbs(A, lbs, ubs, L)
    Omega = init_Omega(weights, biases, D, L)

    a_crown = Omega(L, 1).matmul(weights[1])
    c_crown = sum([Omega(L, i).matmul(biases[i]) for i in range(1, L + 1)]) \
            + sum([Omega(L, i).matmul(weights[i]).matmul(bias_lbs[i - 1]) for i in range(2, L + 1)])

    return gamma * a_crown, gamma * c_crown

def optimize_bound(weights, biases, gamma, alphas, lbs, ubs, thresh, L, layeri, neuron, direction):
    L1 = layeri
    weights1 = weights[:layeri+1]
    biases1  = biases[:layeri+1]
    ubs1 = ubs[:layeri]
    lbs1 = lbs[:layeri]
    alphas1 = alphas[:layeri]

    L2 = L - layeri + 1
    weights2 = [None, torch.eye(weights1[-1].size(0))] + weights[layeri+1:]
    biases2  = [None, torch.zeros(weights1[-1].size(0))] + biases[layeri+1:]
    ubs2 = [ubs[layeri]] + ubs[layeri:]
    lbs2 = [lbs[layeri]] + lbs[layeri:]
    alphas2 = [None] + alphas[layeri:]

    c = torch.zeros(weights2[1].size(1))
    c[neuron] = (1 if direction == "lbs" else -1)

    a_crown_partial, c_crown_partial = get_crown_bounds(weights2, biases2, gamma, alphas2, lbs2, ubs2, L2)

    weights1[-1] = (a_crown_partial + c).matmul(weights1[-1])
    biases1[-1]  = (a_crown_partial + c).matmul(biases1[-1])
    
    a_crown_full, c_crown_full = get_crown_bounds(weights1, biases1, 1.0, alphas1, lbs1, ubs1, L1)
    
    a_crown = a_crown_full
    c_crown = c_crown_partial + c_crown_full

    x_0 = (ubs[0] + lbs[0]) / 2.0
    eps = (ubs[0] - lbs[0]) / 2.0

    return -torch.abs(a_crown).squeeze(0).dot(eps) + a_crown.matmul(x_0) + c_crown + gamma * thresh 

h = torch.Tensor([[-1], [0], [1]])
p = 0.9
thresh = np.log(p / (1 - p))

weights, biases = initialize_weights(trainer.model, h, 3)

lbs = [torch.full((2,), -2.0)]
ubs = [torch.full((2,), 2.0)]
for i in range(1, 3):
    lbs.append(torch.full((weights[i].size(0),), -4.0))
    ubs.append(torch.full((weights[i].size(0),), 4.0))

bounds = {"lbs" : lbs, "ubs" : ubs}


layers = get_num_layers(trainer.model)
direction_layer_pairs = [(direction, layer) for layer in range(layers-1, 0, -1) for direction in ["ubs", "lbs"]]
params_dict = {"lbs" : {}, "ubs" : {}}
for direction, layeri in direction_layer_pairs:
    params_dict[direction][layeri] = {}
    for neuron in range(200):
        gamma, alphas = initialize_params(3, weights)
        params_dict[direction][layeri][neuron] = {'gamma' : gamma, 'alphas' : alphas}


bss = []
cs = [[-0.2326, -1.6094]]

for c in cs:
    bs = []
    for _ in tqdm(range(3), desc="Total iterations"):
        for direction, layeri in tqdm(direction_layer_pairs, desc="Directions & Layers", leave=False):
            neurons = get_num_neurons(trainer.model, layeri - 1) # -1 to account for the input layer
            for neuron in tqdm(range(neurons), desc="Neurons", leave=False):
                gamma = params_dict[direction][layeri][neuron]['gamma']
                alphas = params_dict[direction][layeri][neuron]['alphas']
                optim = torch.optim.SGD([gamma, alphas[1], alphas[2]], lr=0.1, momentum=0.9, maximize=True)
                if direction == "lbs" and (bounds[direction][layeri][neuron] >= 0.0): continue
                if direction == "ubs" and (bounds[direction][layeri][neuron] <= 0.0): continue
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

        m, xs, zs = get_triangle_grb_model(trainer.model, ubs, lbs, h, thresh)
        
        bs.append(get_optimized_grb_result(m, c, zs[0]))
    bss.append(bs)

plot(trainer.model, thresh, cs, bss)