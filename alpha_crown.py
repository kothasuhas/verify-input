from typing import Tuple

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

t = trainer.Trainer(args())
t.load_model("test-weights.pt") # 200 200 3
t.model.eval()

def initialize_weights(model, H, d):
    L = get_num_layers(model)
    weights = [None] + [model[2*i - 1].weight.detach() for i in range(1, L+1)]
    biases = [None] + [model[2*i - 1].bias.detach() for i in range(1, L+1)]

    weights[L] = H.matmul(weights[L])
    biases[L]  = H.matmul(biases[L]) + d

    return weights, biases

def initialize_params(weights, L):
    alphas = [None] + [torch.full((weights[i].size(0),), 0.5, requires_grad=True) for i in range(1, L)]
    gamma = torch.full((weights[-1].size(0), 1), 0.1, requires_grad=True)

    return gamma, alphas

def _get_relu_state_masks(lbs, ubs, A, i):
    relu_on_mask = (lbs[i] >= 0)
    relu_off_mask = (ubs[i] <= 0)
    relu_lower_bound_mask = (~relu_on_mask) & (~relu_off_mask) & (A[i][0] >= 0)
    relu_upper_bound_mask = (~relu_on_mask) & (~relu_off_mask) & (~relu_lower_bound_mask)
    assert torch.all(relu_on_mask ^ relu_off_mask ^ relu_lower_bound_mask ^ relu_upper_bound_mask)
    return relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask

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

        relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask = _get_relu_state_masks(lbs, ubs, A, i)
        D[i][relu_on_mask, relu_on_mask] = 1
        D[i][relu_off_mask,relu_off_mask] = 0
        D[i][relu_lower_bound_mask, relu_lower_bound_mask] = alphas[i][relu_lower_bound_mask]
        D[i][relu_upper_bound_mask, relu_upper_bound_mask] = (ubs[i] / (ubs[i] - lbs[i]))[relu_upper_bound_mask]

    return A, D

def get_bias_lbs(A, lbs, ubs, L):
    bias_lbs = [None]

    for i in range(1, L):
        bias_lbs.append(torch.zeros(A[i].size(1)))
        relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask = _get_relu_state_masks(lbs, ubs, A, i)
        bias_lbs[i][relu_on_mask] = 0
        bias_lbs[i][relu_off_mask] = 0
        bias_lbs[i][relu_lower_bound_mask] = 0
        bias_lbs[i][relu_upper_bound_mask] = (- (ubs[i] * lbs[i]) / (ubs[i] - lbs[i]))[relu_upper_bound_mask]

    return bias_lbs

def init_Omega(weights, biases, D):
    def Omega(end, start):
        assert end >= start
        if end == start: return torch.eye(biases[start].size(0))
        return weights[end].matmul(D[end - 1]).matmul(Omega(end - 1, start))
    return Omega

def get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L):
    A, D = get_diagonals(weights, lbs, ubs, alphas, L)
    bias_lbs = get_bias_lbs(A, lbs, ubs, L)
    Omega = init_Omega(weights, biases, D)

    a_crown = Omega(L, 1).matmul(weights[1])
    c_crown = sum([Omega(L, i).matmul(biases[i]) for i in range(1, L + 1)]) \
            + sum([Omega(L, i).matmul(weights[i]).matmul(bias_lbs[i - 1]) for i in range(2, L + 1)])

    return (a_crown, c_crown) if gamma is None else (gamma.T.matmul(a_crown), gamma.T.matmul(c_crown))

def optimize_bound(weights, biases, gamma, alphas, lbs, ubs, L, layeri, neuron, direction):
    if layeri == 0:
        c = torch.zeros(weights[1].size(1))
        c[neuron] = (1 if direction == "lbs" else -1)
        a_crown, c_crown = get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L)
        a_crown += c

        x_0 = (ubs[0] + lbs[0]) / 2.0
        eps = (ubs[0] - lbs[0]) / 2.0

        return -torch.abs(a_crown).squeeze(0).dot(eps) + a_crown.matmul(x_0) + c_crown 
    else:
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

        a_crown_partial, c_crown_partial = get_crown_bounds(weights2, biases2, gamma, alphas2, lbs2, ubs2, L2)

        c = torch.zeros(weights2[1].size(1))
        c[neuron] = (1 if direction == "lbs" else -1)
        weights1[-1] = (a_crown_partial + c).matmul(weights1[-1])
        biases1[-1]  = (a_crown_partial + c).matmul(biases1[-1])
        
        a_crown_full, c_crown_full = get_crown_bounds(weights1, biases1, None, alphas1, lbs1, ubs1, L1)
        
        a_crown = a_crown_full
        c_crown = c_crown_partial + c_crown_full

        x_0 = (ubs[0] + lbs[0]) / 2.0
        eps = (ubs[0] - lbs[0]) / 2.0

        return -torch.abs(a_crown).squeeze(0).dot(eps) + a_crown.matmul(x_0) + c_crown

def _get_direction_layer_pairs(model: trainer.nn.Sequential):
    num_layers = get_num_layers(model)
    return [(direction, layer) for layer in range(num_layers-1, -1, -1) for direction in ["ubs", "lbs"]]

def initialize_all(model: trainer.nn.Sequential, input_lbs: torch.Tensor, input_ubs: torch.Tensor, H: torch.Tensor, d: torch.Tensor):
    num_layers = get_num_layers(model)
    weights, biases = initialize_weights(model, H, d)

    lbs = [input_lbs]
    ubs = [input_ubs]
    post_activation_lbs = input_lbs
    post_activation_ubs = input_ubs
    for i in range(1, num_layers):
        w = weights[i]
        pre_activation_lbs = torch.where(w > 0, w, 0) @ post_activation_lbs + torch.where(w < 0, w, 0) @ post_activation_ubs + biases[i]
        pre_activation_ubs = torch.where(w > 0, w, 0) @ post_activation_ubs + torch.where(w < 0, w, 0) @ post_activation_lbs + biases[i]
        lbs.append(pre_activation_lbs)
        ubs.append(pre_activation_ubs)
        post_activation_lbs = pre_activation_lbs.clamp(min=0)
        post_activation_ubs = pre_activation_ubs.clamp(min=0)

    L = get_num_layers(t.model)
    params_dict = {"lbs" : {}, "ubs" : {}}
    for direction, layeri in _get_direction_layer_pairs(model):
        params_dict[direction][layeri] = {}
        for neuron in range(get_num_neurons(model, layeri)):
            gamma, alphas = initialize_params(weights, L)
            params_dict[direction][layeri][neuron] = {'gamma' : gamma, 'alphas' : alphas}

    return lbs, ubs, params_dict, weights, biases

bss = []
cs = [[-0.2326, -1.6094]]
H = torch.Tensor([[-1, 0, 1], [-1, 0, 1]])
thresh = np.log(0.9 / (1 - 0.9))
d = torch.Tensor([thresh, thresh])

for c in tqdm(cs, desc="cs"):
    lbs, ubs, params_dict, weights, biases = initialize_all(t.model, torch.Tensor([-2.0, -2.0]), torch.Tensor([2.0, 2.0]), H, d)
    bs = []
    pbar = tqdm(range(6), leave=False)
    for _ in pbar:
        pbar.set_description(f"Best solution: {(-100.0 if len(bs) == 0 else bs[-1])}")
        for direction, layeri in tqdm(_get_direction_layer_pairs(t.model), desc="Directions & Layers", leave=False):
            neurons = get_num_neurons(t.model, layeri)
            for neuron in tqdm(range(neurons), desc="Neurons", leave=False):
                gamma = params_dict[direction][layeri][neuron]['gamma']
                alphas = params_dict[direction][layeri][neuron]['alphas']
                optim = torch.optim.SGD([
                    {'params': gamma, 'lr' : 0.0001}, 
                    {'params': alphas[1]},
                    {'params': alphas[2]}
                ], lr=3.0, momentum=0.9, maximize=True)
                if direction == "lbs" and (lbs[layeri][neuron] >= 0.0) and layeri > 0: continue
                if direction == "ubs" and (ubs[layeri][neuron] <= 0.0) and layeri > 0: continue
                for _ in range(10):
                    optim.zero_grad()
                    loss = optimize_bound(weights, biases, gamma, alphas, lbs, ubs, 3, layeri, neuron, direction)
                    loss.backward()
                    optim.step()

                    with torch.no_grad():
                        if direction == "lbs":
                            lbs[layeri][neuron] = torch.max(lbs[layeri][neuron], loss.detach())
                        else:
                            ubs[layeri][neuron] = torch.min(ubs[layeri][neuron], -loss.detach())
                        gamma.data = torch.clamp(gamma.data, min=0)
                        alphas[1].data = alphas[1].data.clamp(min=0.0, max=1.0)
                        alphas[2].data = alphas[2].data.clamp(min=0.0, max=1.0)

        m, xs, zs = get_triangle_grb_model(t.model, ubs, lbs, H, d)
        
        bs.append(get_optimized_grb_result(m, c, zs[0]))
    bss.append(bs)

plot(t.model, H, d, cs, bss)