from typing import Tuple, List, Dict, Optional

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import torch

import core.trainer as trainer
from .model_utils import get_num_layers, get_num_neurons, get_direction_layer_pairs

import time
import functools
total_time = 0
earliest_call = None
def timeit(func):
    global total_time, earliest_call
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        global total_time, earliest_call
        start_time = time.time()
        if earliest_call is None:
            earliest_call = start_time
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        time_since_first_call = time.time() - earliest_call
        print(f'function [{func.__name__}] finished in {elapsed_time}s (total {total_time}s, first call was {time_since_first_call}s ago)')
        return result
    return new_func



def initialize_weights(
    model: trainer.nn.Sequential,
    H: torch.Tensor,  # (numConstr, 1)
    d: torch.Tensor,  # (numConstr)
) -> Tuple[
    List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    List[Optional[torch.Tensor]],  # [(feat)]
]:
    L = get_num_layers(model)
    weights: List[Optional[torch.Tensor]] = [None] + [model[2*i - 1].weight.detach() for i in range(1, L+1)]  # [(feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]] = [None] + [model[2*i - 1].bias.detach() for i in range(1, L+1)]  # [(feat)]

    weights[L] = H.matmul(weights[L])  # (numConstr, feat_in)
    biases[L]  = H.matmul(biases[L]) + d  # (numConstr)

    return weights, biases

def initialize_params(
    weights: List[Optional[torch.Tensor]], # [(feat_out, feat_in)]
    L: int,
    batch_size: int
) -> Tuple[
    torch.Tensor,  # (batch, numConstr, 1)
    List[Optional[torch.Tensor]],  # [(batch, feat)]
]:
    alphas = [None] + [torch.full((batch_size, weights[i].size(0),), 0.5, requires_grad=True) for i in range(1, L)]
    gamma = torch.full((batch_size, weights[-1].size(0), 1), 0.025, requires_grad=True)  # (batch, numConstr, 1)

    return gamma, alphas

def _interval_bounds(
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
    lbs, ubs = _interval_bounds(num_layers, weights, biases, input_lbs, input_ubs)
    lbs, ubs = tighten_bounds_with_rsip(num_layers, weights, biases, input_lbs, input_ubs, initial_lbs=lbs, initial_ubs=ubs)
    return lbs, ubs


def _relation_to_bounds(
    relation_lbs: torch.Tensor,  # (feat, featInputLayer+1)
    relation_ubs: torch.Tensor,  # (feat, featInputLayer+1)
    input_lbs: torch.Tensor,  # (featInputLayer)
    input_ubs: torch.Tensor,  # (featInputLayer)
) -> Tuple[
    torch.Tensor,  # (feat)
    torch.Tensor,  # (feat)
]:
    ubs = torch.matmul(torch.where(relation_ubs > 0, relation_ubs, 0)[:, :-1], input_ubs)  # (feat)
    ubs += torch.matmul(torch.where(relation_ubs < 0, relation_ubs, 0)[:, :-1], input_lbs)  # (feat)
    ubs += relation_ubs[:, -1]
    lbs = torch.matmul(torch.where(relation_lbs > 0, relation_lbs, 0)[:, :-1], input_lbs)  # (feat)
    lbs += torch.matmul(torch.where(relation_lbs < 0, relation_lbs, 0)[:, :-1], input_ubs)  # (feat)
    lbs += relation_lbs[:, -1]
    assert ubs.dim() == 1
    assert ubs.size(0) == relation_lbs.size(0)
    return lbs, ubs

def tighten_bounds_with_rsip(
    num_layers: int,
    weights: List[torch.Tensor],  # (feat_out, feat_in)
    biases: List[torch.Tensor],  # [(feat)]
    input_lbs: torch.Tensor,  # (featInputLayer)
    input_ubs: torch.Tensor,  # (featInputLayer)
    initial_lbs: List[torch.Tensor],  # (feat)
    initial_ubs: List[torch.Tensor],  # (feat)
    alphas: Optional[List[Optional[torch.Tensor]]] = None,  # List[Optional[torch.Tensor]],  # [(batch, feat)]
) -> Tuple[
    List[torch.Tensor],  # (feat)
    List[torch.Tensor],  # (feat)
]:
    lbs: List[torch.Tensor] = [input_lbs]  # (feat)
    ubs: List[torch.Tensor] = [input_ubs]  # (feat)
    if initial_lbs is not None:
        lbs[0] = torch.max(input_lbs, initial_lbs[0])
        ubs[0] = torch.min(input_ubs, initial_ubs[0])

    relaxations = [None]
    for layeri in range(1, num_layers):
        num_neurons_layeri = weights[layeri].size(0)
        relation_lbs = torch.eye(num_neurons_layeri, num_neurons_layeri + 1)  # (featLayerI, featLayerI + 1)
        relation_ubs = torch.eye(num_neurons_layeri, num_neurons_layeri + 1)  # (featLayerI, featLayerI + 1)
        for layerj in range(layeri, 0, -1):
            w = weights[layerj]  # (featLayer(J+1), featLayerJ)
            b = biases[layerj]

            new_relation_lbs = torch.full((relation_lbs.size(0), w.size(1) + 1), torch.nan)  # (featLayerI, featLayerJ + 1)
            new_relation_lbs[:, :-1] = torch.matmul(relation_lbs[:, :-1], w)  # (featLayerI, featLayerJ)
            new_relation_lbs[:, -1] = relation_lbs[:, -1] + torch.matmul(relation_lbs[:, :-1], b)

            new_relation_ubs = torch.full((relation_ubs.size(0), w.size(1) + 1), torch.nan)  # (featLayerI, featLayerJ + 1)
            new_relation_ubs[:, :-1] = torch.matmul(relation_ubs[:, :-1], w)  # (featLayerI, featLayerJ)
            new_relation_ubs[:, -1] = relation_ubs[:, -1] + torch.matmul(relation_ubs[:, :-1], b)

            if layerj > 1:
                relaxation_w_lbs, relaxation_b_lbs, relaxation_w_ubs, relaxation_b_ubs = relaxations[layerj-1]  # (featLayerJ)

                relation_lbs = new_relation_lbs
                relation_ubs = new_relation_ubs
                new_relation_lbs = torch.full_like(relation_lbs, torch.nan)
                new_relation_ubs = torch.full_like(relation_ubs, torch.nan)

                pos_relation_lbs = torch.where(relation_lbs > 0, relation_lbs, 0)  # (featLayerI, featLayerJ + 1)
                neg_relation_lbs = torch.where(relation_lbs < 0, relation_lbs, 0)  # (featLayerI, featLayerJ + 1)
                pos_relation_ubs = torch.where(relation_ubs > 0, relation_ubs, 0)  # (featLayerI, featLayerJ + 1)
                neg_relation_ubs = torch.where(relation_ubs < 0, relation_ubs, 0)  # (featLayerI, featLayerJ + 1)

                new_relation_lbs[:, :-1] = pos_relation_lbs[:, :-1] * relaxation_w_lbs
                new_relation_lbs[:, :-1] += neg_relation_lbs[:, :-1] * relaxation_w_ubs
                new_relation_ubs[:, :-1] = pos_relation_ubs[:, :-1] * relaxation_w_ubs
                new_relation_ubs[:, :-1] += neg_relation_ubs[:, :-1] * relaxation_w_lbs

                new_relation_lbs[:, -1] = relation_lbs[:, -1] + torch.matmul(pos_relation_lbs[:, :-1], relaxation_b_lbs) + torch.matmul(neg_relation_lbs[:, :-1], relaxation_b_ubs)
                new_relation_ubs[:, -1] = relation_ubs[:, -1] + torch.matmul(pos_relation_ubs[:, :-1], relaxation_b_ubs) + torch.matmul(neg_relation_ubs[:, :-1], relaxation_b_lbs)

            relation_lbs = new_relation_lbs
            relation_ubs = new_relation_ubs

            new_lbs_layeri, new_ubs_layeri = _relation_to_bounds(relation_lbs, relation_ubs, lbs[layerj-1], ubs[layerj-1])
            if layerj == layeri:
                old_lbs_layeri = initial_lbs[layeri]
                old_ubs_layeri = initial_ubs[layeri]
            new_lbs_layeri = torch.max(new_lbs_layeri, old_lbs_layeri)
            new_ubs_layeri = torch.min(new_ubs_layeri, old_ubs_layeri)
            old_lbs_layeri = new_lbs_layeri
            old_ubs_layeri = new_ubs_layeri

        new_lbs, new_ubs = new_lbs_layeri, new_ubs_layeri
        new_relaxations_w_lbs = torch.full(new_lbs.size(), torch.nan)
        new_relaxations_b_lbs = torch.full(new_lbs.size(), torch.nan)
        new_relaxations_w_ubs = torch.full(new_lbs.size(), torch.nan)
        new_relaxations_b_ubs = torch.full(new_lbs.size(), torch.nan)

        relu_off_mask = (new_ubs <= 0)
        new_relaxations_w_lbs[relu_off_mask] = 0
        new_relaxations_b_lbs[relu_off_mask] = 0
        new_relaxations_w_ubs[relu_off_mask] = 0
        new_relaxations_b_ubs[relu_off_mask] = 0

        relu_on_mask = (~relu_off_mask) & (new_lbs >= 0)
        new_relaxations_w_lbs[relu_on_mask] = 1
        new_relaxations_b_lbs[relu_on_mask] = 0
        new_relaxations_w_ubs[relu_on_mask] = 1
        new_relaxations_b_ubs[relu_on_mask] = 0

        relu_unstable_mask = (~relu_off_mask) & (~relu_on_mask)
        positive_dominates_mask = new_ubs > -new_lbs
        if alphas is None:
            new_relaxations_w_lbs[relu_unstable_mask & positive_dominates_mask] = 1
            new_relaxations_b_lbs[relu_unstable_mask & positive_dominates_mask] = 0
            new_relaxations_w_lbs[relu_unstable_mask & ~positive_dominates_mask] = 0
            new_relaxations_b_lbs[relu_unstable_mask & ~positive_dominates_mask] = 0
        else:
            new_relaxations_w_lbs[relu_unstable_mask] = alphas[layeri][relu_unstable_mask]
            new_relaxations_b_lbs[relu_unstable_mask] = 0
            print("Alphas not none")
            exit()

        new_relaxations_w_ubs[relu_unstable_mask] = (new_ubs / (new_ubs - new_lbs))[relu_unstable_mask]
        new_relaxations_b_ubs[relu_unstable_mask] = (-new_lbs * new_ubs / (new_ubs - new_lbs))[relu_unstable_mask]

        relaxations.append((new_relaxations_w_lbs, new_relaxations_b_lbs, new_relaxations_w_ubs, new_relaxations_b_ubs))

        lbs.append(new_lbs)
        ubs.append(new_ubs)
    return lbs, ubs

def initialize_all(
    model: trainer.nn.Sequential,
    input_lbs: torch.Tensor,  # (featInputLayer)
    input_ubs: torch.Tensor,  # (featInputLayer)
    H: torch.Tensor,  # (numConstr, 1)
    d: torch.Tensor,  # (numConstr)
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
        gamma, alphas = initialize_params(weights, L, get_num_neurons(model, layeri))
        params_dict[direction][layeri] = {'gamma' : gamma, 'alphas' : alphas}

    return lbs, ubs, params_dict, weights, biases

def _get_relu_state_masks(
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    A: List[Optional[torch.Tensor]],  # [(batch, feat_outputLayer==1, feat)]
    layeri: int
) -> Tuple[
    torch.Tensor,  # (batch, feat)
    torch.Tensor,  # (batch, feat)
    torch.Tensor,  # (batch, feat)
    torch.Tensor,  # (batch, feat)
]:
    batch_size = A[-1].size(0)
    relu_on_mask = (lbs[layeri] >= 0).tile((batch_size, 1))  # (batch, feat)
    relu_off_mask = (ubs[layeri] <= 0).tile((batch_size, 1))  # (batch, feat)
    a = A[layeri]
    assert a is not None
    assert a.size(1) == 1
    relu_lower_bound_mask: torch.Tensor = (~relu_on_mask) & (~relu_off_mask) & (a[:, 0] >= 0)  # (batch, feat)
    relu_upper_bound_mask: torch.Tensor = (~relu_on_mask) & (~relu_off_mask) & (~relu_lower_bound_mask)  # (batch, feat)
    assert relu_on_mask.dim() == 2
    assert relu_off_mask.dim() == 2
    assert relu_lower_bound_mask.dim() == 2
    assert relu_upper_bound_mask.dim() == 2
    assert len(set(x.shape for x in [relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask])) == 1
    assert torch.all(relu_on_mask ^ relu_off_mask ^ relu_lower_bound_mask ^ relu_upper_bound_mask)
    return relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask

def get_diagonals(
    weights: List[Optional[torch.Tensor]],  # [(batch?, feat_out, feat_in)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    alphas: List[Optional[torch.Tensor]],  # [(batch, feat)]
    L: int
) -> Tuple[
    List[Optional[torch.Tensor]],  # [(batch, feat_outputLayer==1, feat)]
    List[Optional[torch.Tensor]],  # [(batch, feat, feat)]
]:

    A: List[Optional[torch.Tensor]] = [None for _ in range(L)]  # [(batch, feat_outputLayer==1, feat)]
    D: List[Optional[torch.Tensor]] = [None for _ in range(L)]  # [(batch, feat, feat)]
    assert len(weights) == L + 1
    for layeri in range(L-1, 0, -1):  # L-1, ..., 1
        batch_size = alphas[1].size(0)

        if layeri == L-1:
            # Usually, weights[L] has the shape (feat_out, feat_in)
            # Except if it was multiplied by gamma/a_crown_partial in get_crown_bounds/optimize_bounds
            # Then, the batch dimension has already been added
            if weights[L].dim() == 2:
                A[layeri] = weights[L].tile((batch_size, 1, 1))
            else:
                assert weights[L].dim() == 3
                assert weights[L].size(0) == batch_size
                A[layeri] = weights[L]
        else:
            assert weights[layeri+1].dim() == 2  # (feat_out, feat_in)
            A[layeri] = A[layeri+1].matmul(D[layeri+1]).matmul(weights[layeri+1])
        assert A[layeri].dim() == 3
        assert A[layeri].size(1) == 1

        assert weights[layeri].dim() == 2  # (feat_out, feat_in)
        num_feat = weights[layeri].size(0)
        D[layeri] = torch.zeros(batch_size, num_feat, num_feat)

        relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask = _get_relu_state_masks(lbs, ubs, A, layeri)
        assert len(set(x.shape for x in [relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask, ubs[layeri], lbs[layeri]]))  # all (batch, feat)
        assert D[layeri].shape == (batch_size, relu_on_mask.size(1), relu_on_mask.size(1))  # (batch, feat, feat)
        D[layeri][relu_on_mask.diag_embed()] = 1
        D[layeri][relu_off_mask.diag_embed()] = 0
        D[layeri][relu_lower_bound_mask.diag_embed()] = alphas[layeri][relu_lower_bound_mask]
        D[layeri][relu_upper_bound_mask.diag_embed()] = (ubs[layeri] / (ubs[layeri] - lbs[layeri])).tile((batch_size, 1))[relu_upper_bound_mask]

    return A, D

def get_bias_lbs(
    A: List[Optional[torch.Tensor]],  # [(batch, feat_outputLayer==1, feat)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    L: int
) -> List[Optional[torch.Tensor]]:  # [(batch, feat)]

    bias_lbs: List[Optional[torch.Tensor]] = [None]  # [(batch, feat)]

    for i in range(1, L):
        batch_size = A[1].size(0)

        assert A[i] is not None
        num_feat = A[i].size(2)
        bias_lbs.append(torch.zeros(batch_size, num_feat))
        relu_on_mask, relu_off_mask, relu_lower_bound_mask, relu_upper_bound_mask = _get_relu_state_masks(lbs, ubs, A, i)
        bias_lbs[i][relu_on_mask] = 0
        bias_lbs[i][relu_off_mask] = 0
        bias_lbs[i][relu_lower_bound_mask] = 0
        bias_lbs[i][relu_upper_bound_mask] = (- (ubs[i] * lbs[i]) / (ubs[i] - lbs[i])).tile((batch_size, 1))[relu_upper_bound_mask]

    return bias_lbs

def get_Omega(
    weights: List[Optional[torch.Tensor]],  # [(batch?, feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]],  # [(batch?, feat)]
    D: List[Optional[torch.Tensor]],  # [(batch, feat, feat)]
    L: int,
    batch_size: int,  # necessary, as D may be [None], so batch size couldn't be inferred
) -> List[Optional[torch.Tensor]]: # [(batch, feat_outputLayer==1, feat)]
    omegas = [None for _ in range(L+1)]
    for layeri in range(L, 0, -1):
        if layeri == L:
            # Usually, biases[L] has shape (feat)
            # Except if it was multiplied by gamma/a_crown_partial in get_crown_bounds/optimize_bounds
            # Then, the batch dimension has already been added
            if biases[L].dim() == 1:  # (feat)
                omegas[layeri] = torch.eye(biases[L].size(0)).tile((batch_size, 1, 1))
            else:
                assert biases[L].dim() == 2  # (batch, feat)
                assert biases[L].size(0) == batch_size
                omegas[layeri] = torch.eye(biases[L].size(1)).tile((batch_size, 1, 1))
        else:
            # Usually, weights[layer+1] has shape (feat_out, feat_in
            # Except if it was multiplied by gamma/a_crown_partial in get_crown_bounds/optimize_bounds
            # Then, the batch dimension has already been added
            if weights[layeri+1].dim() == 2:  # (feat_out, feat_in)
                omegas[layeri] = omegas[layeri+1] \
                    .matmul(weights[layeri+1].unsqueeze(dim=0)) \
                    .matmul(D[layeri])
            else:
                assert weights[layeri+1].dim() == 3
                assert weights[layeri+1].size(0) == batch_size
                omegas[layeri] = omegas[layeri+1].matmul(weights[layeri+1]).matmul(D[layeri])
    return omegas

def get_crown_bounds(
    weights: List[Optional[torch.Tensor]],  # [(batch?, feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]],  # [(batch?, feat)]
    gamma: Optional[torch.Tensor],  # (batch, numConstr, 1)
    alphas: List[Optional[torch.Tensor]],  # [(batch, feat)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    L: int,
    batch_size: int,  # necessary, as alphas may be [None], so batch size couldn't be inferred
) -> Tuple[
    torch.Tensor,  # (batch, feat_outputLayer==1, featInputLayer)
    torch.Tensor,  # (batch, 1)
]:
    if gamma is not None:
        weights = weights[:-1] + [gamma.mT.matmul(weights[-1])]  # last entry: (batch, 1, featSecondToLastLayer)
        assert weights[-1].dim() == 3
        assert weights[-1].size(0) == batch_size
        assert weights[-1].size(1) == 1
        biases = biases[:-1] + [gamma.mT.matmul(biases[-1])]  # last entry: (batch, 1)
        assert biases[-1].dim() == 2
        assert biases[-1].size(0) == batch_size
        assert biases[-1].size(1) == 1


    A, D = get_diagonals(weights, lbs, ubs, alphas, L)  # [(batch, feat_outputLayer==1, feat)], [(batch, feat, feat)]
    bias_lbs = get_bias_lbs(A, lbs, ubs, L)  # [(batch, feat)]
    Omega = get_Omega(weights, biases, D, L, batch_size)  # [(batch, feat_outputLayer==1, feat)]

    a_crown = Omega[1].matmul(weights[1])  # (batch, feat_outputLayer==1, featInputLayer)
    sum_biases: torch.Tensor = sum([Omega[i].matmul(biases[i].unsqueeze(dim=-1)).squeeze(dim=-1) for i in range(1, L + 1)])  # (batch, 1)
    sum_bias_lbs: torch.Tensor = sum([Omega[i].matmul(weights[i]).matmul(bias_lbs[i - 1].unsqueeze(dim=-1)).squeeze(dim=-1) for i in range(2, L + 1)])  # (batch, 1)
    c_crown = sum_biases + sum_bias_lbs  # (batch, 1)

    assert a_crown.dim() == 3
    assert a_crown.size(0) == batch_size
    assert a_crown.size(1) == 1
    assert a_crown.size(2) == weights[1].size(-1)
    assert c_crown.dim() == 2
    assert c_crown.size(0) == batch_size
    assert c_crown.size(1) == 1
    return (a_crown, c_crown)

def optimize_bound(
    weights: List[Optional[torch.Tensor]],  # [(feat_out, feat_in)]
    biases: List[Optional[torch.Tensor]],  # [(feat)]
    gamma: torch.Tensor,  # (batch, num_constr, 1)
    alphas: List[Optional[torch.Tensor]],  # [(batch, feat)]
    lbs: List[torch.Tensor],  # [(feat)]
    ubs: List[torch.Tensor],  # [(feat)]
    L: int,
    layeri: int,
    direction: str,
) -> torch.Tensor:  # (batch, 1)
    batch_size = gamma.size(0)

    assert weights[0] is None
    for w in weights[1:]:
        assert w is not None
        assert w.dim() == 2

    assert biases[0] is None
    for b in biases[1:]:
        assert b is not None
        assert b.dim() == 1

    assert gamma is not None
    assert gamma.dim() == 3
    assert gamma.size(2) == 1

    assert alphas[0] is None
    for a in alphas[1:]:
        assert a is not None
        assert a.dim() == 2
        assert a.size(0) == batch_size

    for l in lbs:
        assert l is not None
        assert l.dim() == 1
    
    for u in ubs:
        assert u is not None
        assert u.dim() == 1
    
    
    if layeri == 0:
        c = torch.zeros(batch_size, weights[1].size(1))  # (batch, featInputLayer==batch)
        assert weights[1].size(1) == batch_size
        c = (torch.eye(batch_size) if direction == "lbs" else -torch.eye(batch_size))  # (batch, featInputLayer==batch)
        a_crown, c_crown = get_crown_bounds(weights, biases, gamma, alphas, lbs, ubs, L, batch_size)
        # a_crown (batch, 1, featInputLayer)
        # c_crown (batch, 1)
        a_crown += c.unsqueeze(dim=1)  # (batch, 1, featInputLayer)
    else:
        L1 = layeri
        weights1 = weights[:layeri+1]  # [(batch?, feat_out, feat_in)]
        biases1  = biases[:layeri+1]  # [(batch?, feat)]
        ubs1 = ubs[:layeri]  # [(feat)]
        lbs1 = lbs[:layeri]  # [(feat)]
        alphas1 = alphas[:layeri]  # [(batch, feat)]

        L2 = L - layeri + 1
        weights2 = [None, torch.eye(weights1[-1].size(0))] + weights[layeri+1:]  # [(feat_out, feat_in)]
        biases2  = [None, torch.zeros(weights1[-1].size(0))] + biases[layeri+1:]  # [(feat)]
        ubs2 = [ubs[layeri]] + ubs[layeri:]  # [(feat)]
        lbs2 = [lbs[layeri]] + lbs[layeri:]  # [(feat)]
        alphas2 = [None] + alphas[layeri:]  # [(batch, feat)]

        a_crown_partial, c_crown_partial = get_crown_bounds(weights2, biases2, gamma, alphas2, lbs2, ubs2, L2, batch_size)
        # a_crown_partial (batch, 1, featLayerI)
        # c_crown_partial (batch, 1)

        assert batch_size == weights2[1].size(0)
        if direction == "lbs":
            c = torch.eye(batch_size).unsqueeze(dim=1)  # (batch, 1, featLayerI)
        else:
            c = -torch.eye(batch_size).unsqueeze(dim=1)  # (batch, 1, featLayerI)
        weights1[-1] = (a_crown_partial + c).matmul(weights1[-1])  # (batch, 1, featLayerI-1)
        biases1[-1]  = (a_crown_partial + c).matmul(biases1[-1])  # (batch, featLayerI-1)
        
        a_crown_full, c_crown_full = get_crown_bounds(weights1, biases1, None, alphas1, lbs1, ubs1, L1, batch_size)
        # a_crown_full (batch, 1, featInputLayer)
        # c_crown_full (batch, 1)
        
        a_crown = a_crown_full  # (batch, 1, featInputLayer)
        c_crown = c_crown_partial + c_crown_full  # (batch, 1)

    assert a_crown.dim() == 3
    assert a_crown.size(0) == batch_size
    assert a_crown.size(1) == 1
    assert c_crown.dim() == 2
    assert c_crown.size(0) == batch_size
    assert c_crown.size(1) == 1

    x_0 = (ubs[0] + lbs[0]) / 2.0  # (featInputLayer)
    eps = (ubs[0] - lbs[0]) / 2.0  # (featInputLayer)

    res = -torch.abs(a_crown).matmul(eps.unsqueeze(dim=1)).squeeze(dim=2) + a_crown.matmul(x_0.unsqueeze(dim=1)).squeeze(dim=2) + c_crown  # (batch, 1)
    assert res.dim() == 2, res.shape
    assert res.size(0) == batch_size
    assert res.size(1) == 1
    return res
