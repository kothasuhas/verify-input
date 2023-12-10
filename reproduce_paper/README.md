# Reproducing results from the paper

This code was written with the specific experiments from the paper in mind. We do not recommend to use it for other applications.

INVPROP has been integrated into [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN). Please refer to the root directory's README for usage instructions and examples.

## Setup

```
conda create --name invprop python=3.8
source activate invprop
pip install -r requirements.txt
```
## Demos

OOD Demo from paper: `python3 -m ood.ood_demo`

Controls Demo from paper: `python3 -m control.bpset_multistep_demo`

## 
