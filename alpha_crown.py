from typing import List

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from tqdm import tqdm

import core.trainer as trainer


from util.util import get_num_layers, get_num_neurons, plot, get_optimized_grb_result, get_triangle_grb_model, ApproximatedInputBound

class args():
    def __init__(self):
        self.model = "toy"
        self.num_epochs = 1
        self.lr = 0.1

t = trainer.Trainer(args())
t.load_model("test-weights.pt") # 200 200 3
t.model.eval()

def initialize_weights(model, h, thresh):
    L = get_num_layers(model)
    weights = [None]
    biases = [None]
    
    for i in range(1, L+1):
        weights.append(model[2*i - 1].weight.detach())
        biases.append(model[2*i - 1].bias.detach())

    weights[L] = torch.matmul(h.transpose(0, 1), weights[L])
    biases[L]  = torch.matmul(h.transpose(0, 1), biases[L]) + thresh

    return weights, biases

def initialize_params(L, weights):
    alphas = [None]

    for i in range(1, L):
        alphas.append(torch.full((weights[i].size(0),), 0.5))
        alphas[-1].requires_grad = True

    gamma = torch.full((1,), 0.01)
    gamma.requires_grad = True

    return gamma, alphas

def _get_relu_state_masks(lbs, ubs, A, i):
    relu_on_mask = (lbs[i] >= 0)
    relu_off_mask = (~relu_on_mask) * (ubs[i] <= 0)
    relu_lower_bound_mask = (~relu_on_mask) * (~relu_off_mask) * (A[i][0] >= 0)
    relu_upper_bound_mask = (~relu_on_mask) * (~relu_off_mask) * (~relu_lower_bound_mask)
    assert torch.all(torch.logical_xor(torch.logical_xor(torch.logical_xor(relu_on_mask, relu_off_mask), relu_lower_bound_mask), relu_upper_bound_mask))
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

        return -torch.abs(a_crown).squeeze(0).dot(eps) + a_crown.matmul(x_0) + c_crown


def _get_direction_layer_pairs(model: trainer.nn.Sequential):
    num_layers = get_num_layers(model)
    return [(direction, layer) for layer in range(num_layers-1, -1, -1) for direction in ["ubs", "lbs"]]

def initialize_bounds(num_layers: int, weights: List[torch.Tensor], biases: List[torch.Tensor], input_lbs: torch.Tensor, input_ubs: torch.Tensor):
    input_lbs = deepcopy(input_lbs)
    input_ubs = deepcopy(input_ubs)
    lbs = [input_lbs]
    ubs = [input_ubs]
    post_activation_lbs = input_lbs
    post_activation_ubs = input_ubs
    assert len(weights) == num_layers + 1, (len(weights), num_layers)
    for i in range(1, num_layers):
        w = weights[i]
        pre_activation_lbs = torch.where(w > 0, w, 0) @ post_activation_lbs + torch.where(w < 0, w, 0) @ post_activation_ubs + biases[i]
        pre_activation_ubs = torch.where(w > 0, w, 0) @ post_activation_ubs + torch.where(w < 0, w, 0) @ post_activation_lbs + biases[i]
        lbs.append(pre_activation_lbs)
        ubs.append(pre_activation_ubs)
        post_activation_lbs = pre_activation_lbs.clamp(min=0)
        post_activation_ubs = pre_activation_ubs.clamp(min=0)

    return lbs, ubs
        
def initialize_all(model: trainer.nn.Sequential, input_lbs: torch.Tensor, input_ubs: torch.Tensor, h: torch.Tensor, thresh: float):
    num_layers = get_num_layers(model)
    weights, biases = initialize_weights(model, h, thresh)

    lbs, ubs = initialize_bounds(num_layers, weights, biases, input_lbs, input_ubs)

    layers = get_num_layers(t.model)
    params_dict = {"lbs" : {}, "ubs" : {}}
    for direction, layeri in _get_direction_layer_pairs(model):
        params_dict[direction][layeri] = {}
        for neuron in range(get_num_neurons(model, layeri)):
            gamma, alphas = initialize_params(layers, weights)
            params_dict[direction][layeri][neuron] = {'gamma' : gamma, 'alphas' : alphas}

    return lbs, ubs, params_dict, weights, biases

import matplotlib.pyplot as plt
plt.ion()
plt.show()

# Output the Gurobi-Text now
gp.Model()

p = 0.9
thresh = np.log(p / (1 - p))

cs = [[-0.2326, -1.6094]]
cs += [torch.randn(2) for _ in range(2)]
h = torch.Tensor([[-1], [0], [1]])

class InputBranch:
    input_lbs: List[torch.Tensor]
    input_ubs: List[torch.Tensor]
    params_dict: dict
    resulting_lbs: List[torch.Tensor]
    resulting_ubs: List[torch.Tensor]
    weights: List[torch.Tensor]
    biases: List[torch.Tensor]
    
    def __init__(self, input_lbs, input_ubs, params_dict, resulting_lbs, resulting_ubs, weights, biases) -> None:
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs
        self.params_dict = params_dict
        self.resulting_lbs = resulting_lbs
        self.resulting_ubs = resulting_ubs
        self.weights = weights
        self.biases = biases

    def _create_child(self, x_left: bool, y_left: bool):
        x_input_size = self.input_ubs[0] - self.input_lbs[0]
        y_input_size = self.input_ubs[1] - self.input_lbs[1]
        new_x_lbs = self.input_lbs[0] if x_left else self.input_lbs[0] + x_input_size / 2
        new_x_ubs = self.input_lbs[0] + x_input_size / 2 if x_left else self.input_ubs[0]
        new_y_lbs = self.input_lbs[1] if y_left else self.input_lbs[1] + y_input_size / 2
        new_y_ubs = self.input_lbs[1] + y_input_size / 2 if y_left else self.input_ubs[1]

        new_input_lbs = torch.Tensor([new_x_lbs, new_y_lbs])
        new_input_ubs = torch.Tensor([new_x_ubs, new_y_ubs])

        new_resulting_lbs, new_resulting_ubs = initialize_bounds(len(self.weights) - 1, self.weights, self.biases, new_input_lbs, new_input_ubs)
        new_resulting_lbs = [torch.max(x, y) for x, y in zip(new_resulting_lbs, self.resulting_lbs)]
        new_resulting_ubs = [torch.min(x, y) for x, y in zip(new_resulting_ubs, self.resulting_ubs)]
        new_branch = InputBranch(input_lbs=new_input_lbs, input_ubs=new_input_ubs, params_dict=deepcopy(self.params_dict), resulting_lbs=new_resulting_lbs, resulting_ubs=new_resulting_ubs, weights=self.weights, biases=self.biases)

        return new_branch

    def split(self):
        topleft = self._create_child(True, False)
        topright = self._create_child(False, False)
        bottomleft = self._create_child(True, True)
        bottomright = self._create_child(False, True)

        return [topleft, topright, bottomleft, bottomright]
    

approximated_input_bounds: List[ApproximatedInputBound] = []

def get_initial_input_branch(model, h, thresh):
    input_lbs = torch.Tensor([-2.0, -2.0])
    input_ubs = torch.Tensor([2.0, 2.0])
    resulting_lbs, resulting_ubs, params_dict, weights, biases = initialize_all(model=model, input_lbs=input_lbs, input_ubs=input_ubs, h=h, thresh=thresh)

    initial_input_branch = InputBranch(input_lbs=input_lbs, input_ubs=input_ubs, params_dict=params_dict, resulting_lbs=resulting_lbs, resulting_ubs=resulting_ubs, weights=weights, biases=biases)
    return initial_input_branch

branches = [get_initial_input_branch(t.model, h, thresh)]
branches += branches[0].split()


for branch in tqdm(branches, desc="Input Branches"):
    tqdm.write(f"Current input branch area: {branch.input_lbs=}, {branch.input_ubs=}")

    pbar = tqdm(range(3), leave=False)
    last_b = []
    abort = False
    for _ in pbar:
        if abort:
            break
        pbar.set_description(f"Best solution to first bound: {last_b}")
        for direction, layeri in tqdm(_get_direction_layer_pairs(t.model), desc="Directions & Layers", leave=False):
            if abort:
                break
            neurons = get_num_neurons(t.model, layeri)
            for neuron in tqdm(range(neurons), desc="Neurons", leave=False):
                if abort:
                    break
                gamma = branch.params_dict[direction][layeri][neuron]['gamma']
                alphas = branch.params_dict[direction][layeri][neuron]['alphas']
                optim = torch.optim.SGD([
                    {'params': gamma, 'lr' : 0.0001}, 
                    {'params': alphas[1]},
                    {'params': alphas[2]}
                ], lr=3.0, momentum=0.9, maximize=True)
                if direction == "lbs" and (branch.resulting_lbs[layeri][neuron] >= 0.0) and layeri > 0: continue
                if direction == "ubs" and (branch.resulting_ubs[layeri][neuron] <= 0.0) and layeri > 0: continue
                for _ in range(10):
                    optim.zero_grad()
                    loss = optimize_bound(branch.weights, branch.biases, gamma, alphas, branch.resulting_lbs, branch.resulting_ubs, 3, layeri, neuron, direction)
                    loss.backward()
                    optim.step()

                    with torch.no_grad():
                        if direction == "lbs":
                            branch.resulting_lbs[layeri][neuron] = torch.max(branch.resulting_lbs[layeri][neuron], loss.detach())
                        else:
                            branch.resulting_ubs[layeri][neuron] = torch.min(branch.resulting_ubs[layeri][neuron], -loss.detach())
                        if branch.resulting_lbs[layeri][neuron] > branch.resulting_ubs[layeri][neuron]:
                            tqdm.write("[WARNING] Infeasible bounds determined. That's either a bug, or this input region has no intersection with the target area")
                            abort = True
                            break
                        gamma.data = torch.clamp(gamma.data, min=0)
                        alphas[1].data = alphas[1].data.clamp(min=0.0, max=1.0)
                        alphas[2].data = alphas[2].data.clamp(min=0.0, max=1.0)

        if abort:
            break
        m, xs, zs = get_triangle_grb_model(t.model, branch.resulting_ubs, branch.resulting_lbs, h, thresh)
            
        for i, c in tqdm(enumerate(cs), desc="cs", leave=False):
            b = get_optimized_grb_result(m, c, zs[0])
            if i == 0:
                last_b = b
            approximated_input_bounds.append(ApproximatedInputBound(branch.input_lbs, branch.input_ubs, c, b))
        plot(t.model, thresh, approximated_input_bounds)
input("Press any key to terminate")