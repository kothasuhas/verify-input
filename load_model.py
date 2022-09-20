import os, shutil, json, datetime, time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch

import core.trainer as trainer
import core.data as data

from tqdm import tqdm as tqdm

class args():
    def __init__(self):
        self.model = "convnet"
        self.num_epochs = 1
        self.lr = 0.1
        pass

trainer = trainer.Trainer(args())

# trainer.load_model("log/09-17-14:58:58--iso_trial/weights-last.pt")   # width 8
# trainer.load_model("log/09-17-16:30:35--iso_trial/weights-last.pt")   # width 16
trainer.load_model("log/09-17-17:10:13--iso_trial/weights-last.pt")   # width 32
# trainer.load_model("log/09-18-12:33:58--iso_trial/weights-last.pt")   # width 64

conv1 = trainer.model[1]
conv2 = trainer.model[3]
conv3 = trainer.model[5]
conv4 = trainer.model[7]
weight5 = trainer.model[10].weight.detach().numpy()
weight6 = trainer.model[12].weight.detach().numpy()
weight7 = trainer.model[14].weight.detach().numpy()


def conv_kernel(module, input_dim):
    weight = module.weight.detach()
    in_ch = weight.size(1)
    # use identity matrix to get all possible inputs with zeros everywhere except with a 1 at a position of interest
    I = torch.eye(input_dim * input_dim * in_ch, device=weight.device)
    I = I.view(input_dim * input_dim * in_ch, in_ch, input_dim, input_dim)
    # generate kernel by applying intended convolution on previous identity
    conv_kernel = torch.nn.functional.conv2d(I, weight, stride=module.stride, padding=module.padding)
    conv_kernel = conv_kernel.view(input_dim * input_dim * in_ch, -1).t()
    return conv_kernel


def spectral_norm(matrix):
    s = torch.linalg.svdvals(torch.Tensor(matrix))
    return s[0]

lipschitz_constants = [
    spectral_norm(conv_kernel(conv1, 28)) ,
    spectral_norm(conv_kernel(conv2, 28)) ,
    spectral_norm(conv_kernel(conv3, 14)) ,
    spectral_norm(conv_kernel(conv4, 14)) ,
    spectral_norm(weight5) ,
    spectral_norm(weight6) ,
    spectral_norm(weight7)
]

lipschitz_constant = 1
for constant in lipschitz_constants:
    lipschitz_constant *= constant
print(lipschitz_constants)
print(lipschitz_constant)
