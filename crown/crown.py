from typing import Tuple, List, Dict, Optional

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import torch

import core.trainer as trainer
from .model_utils import get_num_layers, get_num_neurons, get_direction_layer_pairs

def initialize_weights(
    model: trainer.nn.Sequential,
    H: torch.Tensor,  # (numConstr==1, 1)
    d: torch.Tensor,  # (numConstr==1)
) -> Tuple[
    List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    List[Optional[torch.Tensor]],  # [(feat)]
]:
    L = get_num_layers(model)
    weights: List[Optional[torch.Tensor]] = [None] + [model[2*i - 1].weight.detach() for i in range(1, L+1)]  # [(feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]] = [None] + [model[2*i - 1].bias.detach() for i in range(1, L+1)]  # [(feat)]

    assert weights[L].size(0) == 1
    weights[L] = H.matmul(weights[L])  # (1, feat_in)
    biases[L]  = H.matmul(biases[L]) + d  # (1)

    return weights, biases

def initialize_params(
    weights: List[Optional[torch.Tensor]], # [(feat_out, feat_in)]
    L: int
) -> Tuple[
    torch.Tensor,  # (1, 1)
    List[Optional[torch.Tensor]],  # [(feat)]
]:
    alphas = [None] + [torch.full((weights[i].size(0),), 0.5, requires_grad=True) for i in range(1, L)]
    assert weights[-1].size(0) == 1
    gamma = torch.full((weights[-1].size(0), 1), 0.1, requires_grad=True)

    return gamma, alphas

def initialize_bounds(
    num_layers: int,
    weights: List[torch.Tensor],  # (feat_out, feat_in)
    biases: List[torch.Tensor],  # [(feat)]
    input_lbs: torch.Tensor,  # (featInputLayer)
    input_ubs: torch.Tensor,  # (featInputLayer)
) -> Tuple[
    List[torch.Tensor],  # (feat)
    List[torch.Tensor],  # (feat)
]:
    input_lbs = deepcopy(input_lbs)
    input_ubs = deepcopy(input_ubs)

    lbs: List[torch.Tensor] = [input_lbs]  # (feat)
    ubs: List[torch.Tensor] = [input_ubs]  # (feat)
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

def initialize_all(
    model: trainer.nn.Sequential,
    input_lbs: torch.Tensor,  # (featInputLayer)
    input_ubs: torch.Tensor,  # (featInputLayer)
    H: torch.Tensor,  # (numConstr==1, 1)
    d: torch.Tensor,  # (numConstr==1)
) -> Tuple[
    List[torch.Tensor],  # [(feat)]
    List[torch.Tensor],  # [(feat)]
    Dict,
    List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    List[Optional[torch.Tensor]],  # [(feat)]
]:
    num_layers = get_num_layers(model)
    weights, biases = initialize_weights(model, H, d)

    lbs, ubs = initialize_bounds(num_layers, weights, biases, input_lbs, input_ubs)

    L = get_num_layers(model)
    
    params_dict = {"lbs" : {}, "ubs" : {}}
    for direction, layeri in get_direction_layer_pairs(model):
        params_dict[direction][layeri] = {}
        for neuron in range(get_num_neurons(model, layeri)):
            gamma, alphas = initialize_params(weights, L)
            params_dict[direction][layeri][neuron] = {'gamma' : gamma, 'alphas' : alphas}

    return lbs, ubs, params_dict, weights, biases

def _get_relu_state_masks(
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    A: List[Optional[torch.Tensor]],  # [(feat_outputLayer==1, feat)]
    layeri: int
) -> Tuple[
    torch.Tensor,  # (feat)
    torch.Tensor,  # (feat)
    torch.Tensor,  # (feat)
    torch.Tensor,  # (feat)
]:
    relu_on_mask = (lbs[layeri] >= 0)  # (feat)
    relu_off_mask = (ubs[layeri] <= 0)  # (feat)
    a = A[layeri]
    assert a.size(0) == 1
    assert a is not None
    relu_lower_bound_mask: torch.Tensor = (~relu_on_mask) & (~relu_off_mask) & (a[0] >= 0)  # (feat)
    relu_upper_bound_mask: torch.Tensor = (~relu_on_mask) & (~relu_off_mask) & (~relu_lower_bound_mask)  # (feat)
    assert relu_on_mask.dim() == 1
    assert relu_off_mask.dim() == 1
    assert relu_lower_bound_mask.dim() == 1
    assert relu_upper_bound_mask.dim() == 1
    assert len(set(x.shape for x in [relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask])) == 1
    assert torch.all(relu_on_mask ^ relu_off_mask ^ relu_lower_bound_mask ^ relu_upper_bound_mask)
    return relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask

def get_diagonals(
    weights: List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    alphas: List[Optional[torch.Tensor]],  # [(feat)]
    L: int
) -> Tuple[
    List[Optional[torch.Tensor]],  # [(feat_outputLayer==1, feat)]
    List[Optional[torch.Tensor]],  # [(feat, feat)]
]:
    A: List[Optional[torch.Tensor]] = [None for _ in range(L)]  # [(feat_outputLayer==1, feat)]
    D: List[Optional[torch.Tensor]] = [None for _ in range(L)]  # [(feat, feat)]
    assert len(weights) == L + 1
    for layeri in range(L-1, 0, -1):  # L-1, ..., 1
        if layeri == L-1:
            A[layeri] = weights[L]
        else:
            A[layeri] = A[layeri+1].matmul(D[layeri+1]).matmul(weights[layeri+1])
        assert A[layeri].dim() == 2
        assert A[layeri].size(0) == 1

        num_feat = weights[layeri].size(0)
        D[layeri] = torch.zeros(num_feat, num_feat)

        relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask = _get_relu_state_masks(lbs, ubs, A, layeri)
        assert len(set(x.shape for x in [relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask, ubs[layeri], lbs[layeri]]))  # all (feat)
        assert D[layeri].shape == (relu_on_mask.size(0), relu_on_mask.size(0))
        D[layeri][relu_on_mask, relu_on_mask] = 1
        D[layeri][relu_off_mask,relu_off_mask] = 0
        D[layeri][relu_lower_bound_mask, relu_lower_bound_mask] = alphas[layeri][relu_lower_bound_mask]
        D[layeri][relu_upper_bound_mask, relu_upper_bound_mask] = (ubs[layeri] / (ubs[layeri] - lbs[layeri]))[relu_upper_bound_mask]

    return A, D

def get_bias_lbs(
    A: List[Optional[torch.Tensor]],  # [(feat_outputLayer==1, feat)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    L: int
) -> List[Optional[torch.Tensor]]:  # [(feat)]
    bias_lbs: List[Optional[torch.Tensor]] = [None]  # [(feat)]

    for i in range(1, L):
        assert A[i] is not None
        num_feat = A[i].size(1)
        bias_lbs.append(torch.zeros(num_feat))
        relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask = _get_relu_state_masks(lbs, ubs, A, i)
        bias_lbs[i][relu_on_mask] = 0
        bias_lbs[i][relu_off_mask] = 0
        bias_lbs[i][relu_lower_bound_mask] = 0
        bias_lbs[i][relu_upper_bound_mask] = (- (ubs[i] * lbs[i]) / (ubs[i] - lbs[i]))[relu_upper_bound_mask]

    return bias_lbs

def get_Omega(
    weights: List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]],  # [(feat)]
    D: List[Optional[torch.Tensor]],  # [(feat, feat)]
    L: int
) -> List[Optional[torch.Tensor]]: # [(feat_outputLayer==1, feat)]
    omegas = [None for _ in range(L+1)]
    for layeri in range(L, 0, -1):
        if layeri == L:
            omegas[layeri] = torch.eye(biases[L].size(0))
        else:
            omegas[layeri] = omegas[layeri+1].matmul(weights[layeri+1]).matmul(D[layeri])
    return omegas

def get_crown_bounds(
    weights: List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]],  # [(feat)]
    gamma: torch.Tensor,  # (numConstr==1, 1)
    alphas: List[Optional[torch.Tensor]],  # [(feat)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    L: int,
) -> Tuple[
    torch.Tensor,  # (feat_outputLayer==1, featInputLayer)
    torch.Tensor,  # (1)
]:
    A, D = get_diagonals(weights, lbs, ubs, alphas, L)  # [(feat_outputLayer==1, feat)], [(feat, feat)]
    bias_lbs = get_bias_lbs(A, lbs, ubs, L)  # [(feat)]
    Omega = get_Omega(weights, biases, D, L)  # [(feat_outputLayer==1, feat)]

    a_crown = Omega[1].matmul(weights[1])  # (feat_outputLayer==1, featInputLayer)
    sum_biases: torch.Tensor = sum([Omega[i].matmul(biases[i]) for i in range(1, L + 1)])  # (1)
    sum_bias_lbs: torch.Tensor = sum([Omega[i].matmul(weights[i]).matmul(bias_lbs[i - 1]) for i in range(2, L + 1)])  # (1)
    c_crown = sum_biases + sum_bias_lbs  # (1)

    if gamma is not None:
        a_crown = gamma.T.matmul(a_crown)  # (1, featInputLayer)
        c_crown = gamma.T.matmul(c_crown)  # (1)
    return (a_crown, c_crown)

def optimize_bound(
    weights: List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]],  # [(feat)]
    gamma: torch.Tensor,  # (num_constr==1, 1)
    alphas: List[Optional[torch.Tensor]],  # [(feat)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    L: int,
    layeri: int,
    neuron: int,
    direction: str,
) -> torch.Tensor:  # (1)
    assert weights[0] is None
    for w in weights[1:]:
        assert w is not None
        assert w.dim() == 2

    assert biases[0] is None
    for b in biases[1:]:
        assert b is not None
        assert b.dim() == 1

    assert gamma is not None
    assert gamma.dim() == 2
    assert gamma.size(0) == 1
    assert gamma.size(1) == 1

    assert alphas[0] is None
    for a in alphas[1:]:
        assert a is not None
        assert a.dim() == 1

    for l in lbs:
        assert l is not None
        assert l.dim() == 1
    
    for u in ubs:
        assert u is not None
        assert u.dim() == 1
    
    
    if layeri == 0:
        c = torch.zeros(weights[1].size(1))
        c[neuron] = (1 if direction == "lbs" else -1)
        a_crown, c_crown = get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L)
        a_crown += c  # (1, featInputLayer)

        x_0 = (ubs[0] + lbs[0]) / 2.0  # (featInputLayer)
        eps = (ubs[0] - lbs[0]) / 2.0  # (featInputLayer)

        return -torch.abs(a_crown).matmul(eps) + a_crown.matmul(x_0) + c_crown  # (1)
    else:
        L1 = layeri
        weights1 = weights[:layeri+1]  # [(feat_out, feat_in)]
        biases1  = biases[:layeri+1]  # [(feat)]
        ubs1 = ubs[:layeri]  # [(feat)]
        lbs1 = lbs[:layeri]  # [(feat)]
        alphas1 = alphas[:layeri]  # [(feat)]

        L2 = L - layeri + 1
        weights2 = [None, torch.eye(weights1[-1].size(0))] + weights[layeri+1:]  # [(feat_out, feat_in)]
        biases2  = [None, torch.zeros(weights1[-1].size(0))] + biases[layeri+1:]  # [(feat)]
        ubs2 = [ubs[layeri]] + ubs[layeri:]  # [(feat)]
        lbs2 = [lbs[layeri]] + lbs[layeri:]  # [(feat)]
        alphas2 = [None] + alphas[layeri:]  # [(feat)]

        a_crown_partial, c_crown_partial = get_crown_bounds(weights2, biases2, gamma, alphas2, lbs2, ubs2, L2)
        # a_crown_partial (1, featLayerI)
        # c_crown_partial (1)

        c = torch.zeros((1, weights2[1].size(1)))  # (1, featLayerI)
        c[0, neuron] = (1 if direction == "lbs" else -1)
        weights1[-1] = (a_crown_partial + c).matmul(weights1[-1])  # (1, featLayerI-1)
        biases1[-1]  = (a_crown_partial + c).matmul(biases1[-1])  # (featLayerI-1)
        
        a_crown_full, c_crown_full = get_crown_bounds(weights1, biases1, None, alphas1, lbs1, ubs1, L1)
        # a_crown_full (1, featInputLayer)
        # c_crown_full (1)
        
        a_crown = a_crown_full  # (1, featInputLayer)
        c_crown = c_crown_partial + c_crown_full  # (1)

        x_0 = (ubs[0] + lbs[0]) / 2.0  # (featInputLayer)
        eps = (ubs[0] - lbs[0]) / 2.0  # (featInputLayer)

        return -torch.abs(a_crown).matmul(eps) + a_crown.matmul(x_0) + c_crown  # (1)

class ApproximatedInputBound:
    input_lbs: List[float]
    input_ubs: List[float]
    c: List[float]
    b: float

    def __init__(self, input_lbs, input_ubs, c, b) -> None:
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs
        self.c = c
        self.b = b


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
