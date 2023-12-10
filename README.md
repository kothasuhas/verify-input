# INVPROP

This repository provides code to reproduce the results from the paper [Provably Bounding Neural Network Preimages](https://arxiv.org/abs/2302.01404) and examples on how to apply this technique to other usecases.

<p align="center">
<img src="https://user-images.githubusercontent.com/38450656/216413863-9a1d2422-94cc-4f4f-b0fe-c40ec4dcbbb9.png" width=400/>
</p>

## Reproduction of results in the paper

To reproduce the results from the paper, please use the code in `reproduce_paper`. This code was written with these specific experiments in mind. We do not recommend to use it for other applications.

# Applying INVPROP to other applications

INVPROP has been integrated into [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN). This enables the application of INVPROP to other applications.

<p align="center">
<img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="36%">
</p>

## Setup

To use INVPROP, you only need to install [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN).
Note: The updated version that includes INVPROP will be published within the next few days. We will remove this note once the update is published.
Please refer to the alpha-beta-CROWN repository for installation instructions.

## Usage

Follow these steps to use INVPROP. Note that this repository contains examples that may help you.

1. In a preprocessing script, load the model that should be used and add an additional fully connected layer to the front of the model. For N input neurons that layer should have N+C neurons to define C additional c. The weight for this layer must be a unit-matrix for the first N neurons and some linear combination of inputs for the additional C neurons. These combinations define the cs. For each such c, both an upper and a lower bound will be computed (in our plots, this corresponds to two lines, on opposite sides of the bounded input region). Export this modified network as a new onnx model.
2. Define a VNN-LIB file just like for forward verification. The output constraint defines what should be used to bound the input. If this is a disjunction (e.g. in the OOD benchmark, there are two output regions, y1>y3 and y2>y3), then for each region the inputs will be bounded separately. This means we're branching over the output constraints by default. To avoid this, you'd need to adapt the onnx model again, and e.g. add additional layers that encode max(y1,y2), so only one output constraint remains. But there's no clear benefit in doing so.
3. Call alpha-beta-CROWN using the modified onnx file, and look for the log output at the beginning:

```
Model: BoundedModule(
  (/0): BoundInput(name=/0, inputs=[], perturbed=True)
  (/9): BoundParams(name=/9, inputs=[], perturbed=False)
  (/10): BoundParams(name=/10, inputs=[], perturbed=False)
  (/11): BoundParams(name=/11, inputs=[], perturbed=False)
  (/12): BoundParams(name=/12, inputs=[], perturbed=False)
  (/13): BoundParams(name=/13, inputs=[], perturbed=False)
  (/14): BoundParams(name=/14, inputs=[], perturbed=False)
  (/15): BoundParams(name=/15, inputs=[], perturbed=False)
  (/16): BoundParams(name=/16, inputs=[], perturbed=False)
  (/17): BoundFlatten(name=/17, inputs=[/0], perturbed=True)
  (/18): BoundLinear(name=/18, inputs=[/17, /9, /10], perturbed=True)
  (/input): BoundLinear(name=/input, inputs=[/18, /11, /12], perturbed=True)
  (/20): BoundRelu(name=/20, inputs=[/input], perturbed=True)
  (/input.3): BoundLinear(name=/input.3, inputs=[/20, /13, /14], perturbed=True)
  (/22): BoundRelu(name=/22, inputs=[/input.3], perturbed=True)
  (/23): BoundLinear(name=/23, inputs=[/22, /15, /16], perturbed=True)
)
Original output: tensor([[ 0.62595803, -8.65573406,  8.08960819]], device='cuda:0')
Split layers:
  BoundLinear(name=/input, inputs=[/18, /11, /12], perturbed=True): [(BoundRelu(name=/20, inputs=[/input], perturbed=True), 0)]
  BoundLinear(name=/input.3, inputs=[/20, /13, /14], perturbed=True): [(BoundRelu(name=/22, inputs=[/input.3], perturbed=True), 0)]
Nonlinear functions:
   BoundRelu(name=/20, inputs=[/input], perturbed=True)
   BoundRelu(name=/22, inputs=[/input.3], perturbed=True)
```

This tells us that the linear layer added in step 1 is called '/18', and that the layers '/input' and '/input.3' are used for ReLU relaxations.

4. Define a simple script that processes the tightened input bounds once they are computed, e.g.:

```
import sys
from abcrown import ABCROWN

if __name__ == '__main__':
    abcrown = ABCROWN(args=sys.argv[1:] + ['--return_optimized_model'])
    model = abcrown.main()

    print(model.net['/18'].lower)
    print(model.net['/18'].upper)
```

5. Run this e.g. like `PYTHONPATH=$PYTHONPATH:/path/to/alpha-beta-CROWN/complete_verifier/ python oc.py --config path/to/ood.yaml --onnx_path path/to/ood.onnx --vnnlib_path path/to/ood.vnnlib --apply_output_constraints_to /input /input.3 /18   --optimize_disjuncts_separately --tighten_input_bounds --directly_optimize /18 --oc_lr 0.01`

- `PYTHONPATH=$PYTHONPATH:/path/to/alpha-beta-CROWN/complete_verifier/` (adapt to your path) allows to call abCROWN.
- `oc.py` is the name of the script created in step 4. It passes all arguments on to abCROWN, so the same set of arguments can be used.
- `--apply_output_constraints_to /input /input.3 /18` activates output constraints for these three layers. You could also use `--apply_output_constraints_to BoundLinear` to simply apply them to every linear layer. Make sure to list the name of the layer added in step 1.
- `--optimize_disjuncts_separately` must be set if the output constraint is a disjunction.
- `--tighten_input_bounds` will tighten the input bounds. Output constraints are automatically applied to this layer.
- `--directly_optimize /18` must be the name of the layer added in step 1.
- `--oc_lr 0.01` the learning rate for the gammas introduced in the dualization of the output constraints.

6. Once alpha-beta-CROWN is done, the returned model can be used to access the bounds: `model.net['/18'].lower` This will be of shape ` [num_output_constraints, N+C]`.

## Examples

For examples on how to use alpha-beta-CROWN to reproduce the experiments in our paper, please refer to the `examples` directory.

## Support

If you have questions about the use of INVPROP in alpha-beta-CROWN, please reach out to Christopher Brix (brix@cs.rwth-aachen.de).

# Citation

If you find this code useful, please consider citing our NeurIPS paper:

```
@article{zhang2018efficient,
  title={Provably Bounding Neural Network Preimages},
  author={Kotha, Suhas and Brix, Christopher and Kolter, J Zico and Dvijotham, Krishnamurthy Dj and Zhang, Huan},
  journal={Advances in Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/pdf/2302.01404.pdf}
}
```