from typing import List, Optional, Tuple
import sys
import torch
import numpy as np
from plotting_utils.plotting import ApproximatedInputBound, plot2d
from abcrown import ABCROWN

def plot_iteration(model, first_layer_name, plot_name):
    H = torch.Tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    d = torch.Tensor([4.5, -5.0, -0.25, -0.25])
    # Note that the same cs are defined in the prepare.py script!
    num_cs = 20
    cs = torch.tensor([[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)])
    cs = torch.where(torch.abs(cs) > 0.0001, cs, 0.0001)

    halfspaces = [
        ApproximatedInputBound(
            torch.tensor([-5., -5.]).cpu(),
            torch.tensor([5., 5.]).cpu(),
            cs[i].numpy(),
            model.net[first_layer_name].lower[0,i+2].cpu().numpy()
        ) for i in range(num_cs)
    ]
    halfspaces += [
        ApproximatedInputBound(
            torch.tensor([-5., -5.]).cpu(),
            torch.tensor([5., 5.]).cpu(),
            -cs[i].numpy(),
            -model.net[first_layer_name].upper[0,i+2].cpu().numpy()
        ) for i in range(num_cs)
    ]
    plot2d(
        model=model,
        H=H,
        d=d,
        constraint_merger=torch.all,
        approximated_input_bounds=halfspaces,
        input_lbs=[-5., -5.],
        input_ubs=[5., 5.],
        branch=None,
        name=plot_name,
    )

if __name__ == '__main__':
    # plotting will perform a forward pass on the model, which will delete the .lower and .upper parameters
    # so we'll need to save them first, before calling plot_iteration

    first_layer_name = '/26'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_1.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main()
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_1.png')

    first_layer_name = '/37'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_2.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_2.png')

    first_layer_name = '/48'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_3.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_3.png')

    first_layer_name = '/59'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_4.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_4.png')

    first_layer_name = '/70'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_5.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
        "/input.39": (model.net["/input.31"].lower, model.net["/input.31"].upper),
        "/input.43": (model.net["/input.35"].lower, model.net["/input.35"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_5.png')

    first_layer_name = '/81'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_6.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
        "/input.39": (model.net["/input.31"].lower, model.net["/input.31"].upper),
        "/input.43": (model.net["/input.35"].lower, model.net["/input.35"].upper),
        "/input.47": (model.net["/input.39"].lower, model.net["/input.39"].upper),
        "/input.51": (model.net["/input.43"].lower, model.net["/input.43"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_6.png')

    first_layer_name = '/92'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_7.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
        "/input.39": (model.net["/input.31"].lower, model.net["/input.31"].upper),
        "/input.43": (model.net["/input.35"].lower, model.net["/input.35"].upper),
        "/input.47": (model.net["/input.39"].lower, model.net["/input.39"].upper),
        "/input.51": (model.net["/input.43"].lower, model.net["/input.43"].upper),
        "/input.55": (model.net["/input.47"].lower, model.net["/input.47"].upper),
        "/input.59": (model.net["/input.51"].lower, model.net["/input.51"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_7.png')

    first_layer_name = '/103'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_8.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
        "/input.39": (model.net["/input.31"].lower, model.net["/input.31"].upper),
        "/input.43": (model.net["/input.35"].lower, model.net["/input.35"].upper),
        "/input.47": (model.net["/input.39"].lower, model.net["/input.39"].upper),
        "/input.51": (model.net["/input.43"].lower, model.net["/input.43"].upper),
        "/input.55": (model.net["/input.47"].lower, model.net["/input.47"].upper),
        "/input.59": (model.net["/input.51"].lower, model.net["/input.51"].upper),
        "/input.63": (model.net["/input.55"].lower, model.net["/input.55"].upper),
        "/input.67": (model.net["/input.59"].lower, model.net["/input.59"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_8.png')

    first_layer_name = '/114'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_9.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    next_interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
        "/input.39": (model.net["/input.31"].lower, model.net["/input.31"].upper),
        "/input.43": (model.net["/input.35"].lower, model.net["/input.35"].upper),
        "/input.47": (model.net["/input.39"].lower, model.net["/input.39"].upper),
        "/input.51": (model.net["/input.43"].lower, model.net["/input.43"].upper),
        "/input.55": (model.net["/input.47"].lower, model.net["/input.47"].upper),
        "/input.59": (model.net["/input.51"].lower, model.net["/input.51"].upper),
        "/input.63": (model.net["/input.55"].lower, model.net["/input.55"].upper),
        "/input.67": (model.net["/input.59"].lower, model.net["/input.59"].upper),
        "/input.71": (model.net["/input.63"].lower, model.net["/input.63"].upper),
        "/input.75": (model.net["/input.67"].lower, model.net["/input.67"].upper),
    }
    plot_iteration(model, first_layer_name, 'results/plot_9.png')

    first_layer_name = '/125'
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "files/double_integrator_10.onnx", "--directly_optimize", first_layer_name])
    model = abcrown.main(interm_bounds=next_interm_bounds)
    plot_iteration(model, first_layer_name, 'results/plot_10.png')
