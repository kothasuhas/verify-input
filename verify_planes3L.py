from typing import List

import gurobipy as gp
import numpy as np
import torch
from tqdm import tqdm

import core.trainer as trainer

from crown.lp import plot, get_optimal_grb_model, get_optimized_grb_result


def optimal_grb(model: trainer.nn.Sequential, h: torch.Tensor, thresh: float, cs: List[torch.Tensor]):
    bs = []
    try:
        m, xs, zs = get_optimal_grb_model(model, h, thresh)
        for c in tqdm(cs):
            bs.append(get_optimized_grb_result(m, c, zs[0]))
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    return bs



def main():
    class args():
        def __init__(self):
            self.model = "toy"
            self.num_epochs = 1
            self.lr = 0.1

    t = trainer.Trainer(args())
    t.load_model("test-weights.pt") # 200 200 3

    # cs = [torch.randn(2) for _ in range(3)]
    #cs = []
    cs = [[-0.2326, -1.6094]]

    p = 0.90
    thresh = np.log(p / (1 - p))

    h = torch.Tensor([[-1], [0], [1]])
    bs = optimal_grb(t.model, h, thresh, cs)

    plot(t.model, thresh, cs, [bs])

    

main()