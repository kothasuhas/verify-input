import sys
from abcrown import ABCROWN
import time

if __name__ == '__main__':
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_1.onnx", "--directly_optimize", "/21"])
    model = abcrown.main()

    interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
    }
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_2.onnx", "--directly_optimize", "/29"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
    }
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_3.onnx", "--directly_optimize", "/37"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
    }
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_4.onnx", "--directly_optimize", "/45"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
        "/input.7": (model.net["/input"].lower, model.net["/input"].upper),
        "/input.11": (model.net["/input.3"].lower, model.net["/input.3"].upper),
        "/input.15": (model.net["/input.7"].lower, model.net["/input.7"].upper),
        "/input.19": (model.net["/input.11"].lower, model.net["/input.11"].upper),
        "/input.23": (model.net["/input.15"].lower, model.net["/input.15"].upper),
        "/input.27": (model.net["/input.19"].lower, model.net["/input.19"].upper),
        "/input.31": (model.net["/input.23"].lower, model.net["/input.23"].upper),
        "/input.35": (model.net["/input.27"].lower, model.net["/input.27"].upper),
    }
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_5.onnx", "--directly_optimize", "/53"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
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
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_6.onnx", "--directly_optimize", "/61"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
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
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_7.onnx", "--directly_optimize", "/69"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
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
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_8.onnx", "--directly_optimize", "/77"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
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
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_9.onnx", "--directly_optimize", "/85"])
    model = abcrown.main(interm_bounds=interm_bounds)

    interm_bounds = {
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
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model', "--onnx_path", "double_integrator_10.onnx", "--directly_optimize", "/93"])
    model = abcrown.main(interm_bounds=interm_bounds)

    print(model.net["/93"].lower)
    print(model.net["/93"].upper)