from typing import List, Optional, Tuple
import sys
import torch
import numpy as np
from plotting_utils.plotting import ApproximatedInputBound, plot2d
from abcrown import ABCROWN


if __name__ == '__main__':
    abcrown = ABCROWN(args=sys.argv[1:])
    model = abcrown.main()

    H = torch.Tensor([[-1., 0., 1.], [0., -1., 1.]])
    d = torch.Tensor([0.0, 0.0])

    # Note that the same cs are defined in the prepare.py script!
    num_cs = 20
    cs = torch.tensor([[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)])
    cs = torch.where(torch.abs(cs) > 0.0001, cs, 0.0001)

    halfspaces = [
        ApproximatedInputBound(
            torch.tensor([-2., -2.]).cpu(),
            torch.tensor([0., 0.]).cpu(),
            cs[i].numpy(),
            model.net['/21'].lower[0,i+2].cpu().numpy()
        ) for i in range(num_cs)
    ]
    halfspaces += [
        ApproximatedInputBound(
            torch.tensor([-2., -2.]).cpu(),
            torch.tensor([0., 0.]).cpu(),
            -cs[i].numpy(),
            -model.net['/21'].upper[0,i+2].cpu().numpy()
        ) for i in range(num_cs)
    ]

    plot2d(
        model=model,
        H=H,
        d=d,
        constraint_merger=torch.any,
        approximated_input_bounds=halfspaces,
        input_lbs=[-2., -2.],
        input_ubs=[2., 2.],
        branch=None,
        name="results/plot.png"
    )
