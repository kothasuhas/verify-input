# INVPROP

WIP Repo for the paper [Provably Bounding Neural Network Preimages](https://arxiv.org/abs/2302.01404). We plan on merging these functionalities into the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) library. This current repo only supports the core bare functionality.

<p align="center">
<img src="https://user-images.githubusercontent.com/38450656/216413863-9a1d2422-94cc-4f4f-b0fe-c40ec4dcbbb9.png" width=400/>
</p>

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
