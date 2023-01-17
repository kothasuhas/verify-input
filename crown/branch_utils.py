from typing import List, Optional

from copy import deepcopy
import torch

from crown.crown import initialize_bounds

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

class ExcludedInputRegions:
    input_lbs: List[torch.Tensor]
    input_ubs: List[torch.Tensor]

    def __init__(self, input_lbs, input_ubs) -> None:
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs

class InputBranch:
    input_lbs: List[torch.Tensor]
    input_ubs: List[torch.Tensor]
    params_dict: dict
    resulting_lbs: List[torch.Tensor]
    resulting_ubs: List[torch.Tensor]
    weights: List[torch.Tensor]
    biases: List[torch.Tensor]
    remaining_max_branching_depth: int
    last_b_sum: Optional[float]
    optimizers: List[torch.optim.SGD]
    
    def __init__(
        self,
        input_lbs,
        input_ubs,
        params_dict,
        resulting_lbs,
        resulting_ubs,
        weights,
        biases,
        remaining_max_branching_depth,
        last_b_sum: Optional[float] = None,
        old_optimizers: Optional[List[torch.optim.SGD]] = None
    ) -> None:
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs
        self.params_dict = params_dict
        self.resulting_lbs = resulting_lbs
        self.resulting_ubs = resulting_ubs
        self.weights = weights
        self.biases = biases
        self.remaining_max_branching_depth = remaining_max_branching_depth
        self.last_b_sum = last_b_sum
        self.optimizers = []
        for layeri in range(len(self.biases) - 1):
            opti = torch.optim.SGD([
                {'params': params_dict[layeri]['gamma'], 'lr' : 0.001},
                {'params': params_dict[layeri]['alphas'][1:]},
            ], lr=3.0, momentum=0.9, maximize=True)
            if old_optimizers is not None:
                s = old_optimizers[layeri].state_dict()
                opti.load_state_dict(s)
            self.optimizers.append(opti)


    def _create_child(self, x_left: bool, y_left: bool):
        x_input_size = self.input_ubs[0] - self.input_lbs[0]
        y_input_size = self.input_ubs[1] - self.input_lbs[1]
        new_x_lbs = self.input_lbs[0] if x_left else self.input_lbs[0] + x_input_size / 2
        new_x_ubs = self.input_lbs[0] + x_input_size / 2 if x_left else self.input_ubs[0]
        new_y_lbs = self.input_lbs[1] if y_left else self.input_lbs[1] + y_input_size / 2
        new_y_ubs = self.input_lbs[1] + y_input_size / 2 if y_left else self.input_ubs[1]

        new_input_lbs = torch.Tensor([new_x_lbs, new_y_lbs])
        new_input_ubs = torch.Tensor([new_x_ubs, new_y_ubs])

        new_resulting_lbs, new_resulting_ubs = initialize_bounds(
            num_layers=len(self.weights) - 1,
            weights=self.weights,
            biases=self.biases,
            input_lbs=new_input_lbs,
            input_ubs=new_input_ubs,
            initial_lbs=self.resulting_lbs,
            initial_ubs=self.resulting_ubs
        )

        new_branch = InputBranch(
            input_lbs=new_input_lbs,
            input_ubs=new_input_ubs,
            params_dict=deepcopy(self.params_dict),
            resulting_lbs=new_resulting_lbs,
            resulting_ubs=new_resulting_ubs,
            weights=self.weights,
            biases=self.biases,
            remaining_max_branching_depth=None if self.remaining_max_branching_depth is None else self.remaining_max_branching_depth - 1,
            last_b_sum=self.last_b_sum,
            old_optimizers=self.optimizers,
        )

        return new_branch

    def split(self):
        topleft = self._create_child(True, False)
        topright = self._create_child(False, False)
        bottomleft = self._create_child(True, True)
        bottomright = self._create_child(False, True)

        return [topleft, topright, bottomleft, bottomright]