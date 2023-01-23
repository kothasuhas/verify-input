from enum import Enum
from typing import List, Optional

import numpy as np
import gurobipy as gp
import math
import torch
from torch.autograd import Variable

import core.trainer as trainer
import matplotlib.pyplot as plt

from crown.model_utils import load_model

plt.rcParams["figure.figsize"] = (3, 3)
plt.cla()

MIN_X_INPUT_VALUE = -2.0
MIN_Y_INPUT_VALUE = -2.0
MAX_X_INPUT_VALUE = 2.0
MAX_Y_INPUT_VALUE = 2.0

p = 0.9

model = load_model("toy", "log/01-22-17:31:27--bad_contour/weights-last.pt")

resolution_x = 1000
resolution_y = 1000
remaining_input_area = np.ones((resolution_y, resolution_x))
XX, YY = np.meshgrid(np.linspace(MIN_X_INPUT_VALUE, MAX_X_INPUT_VALUE, resolution_x), np.linspace(MIN_Y_INPUT_VALUE, MAX_Y_INPUT_VALUE, resolution_y))
X0 = Variable(torch.tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T, device="cpu", dtype=torch.float32))
# https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently#comment104116008_58926343
orig_model_location = next(model.parameters()).device
model.to("cpu")
y0 = model(X0)
model.to(orig_model_location)

id = torch.abs(y0[:,1] - y0[:,0])
id_range = torch.max(id) - torch.min(id)
id = - id + id_range/4.0
ZZ = id.resize(resolution_y,resolution_x).data.numpy()
bound = max(np.abs(np.min(ZZ)), np.max(ZZ)) + 1
plt.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-bound, bound, 30))

plt.axis("equal")

plt.xlim(MIN_X_INPUT_VALUE - 0.1, MAX_X_INPUT_VALUE + 0.1)
plt.ylim(MIN_Y_INPUT_VALUE - 0.1, MAX_Y_INPUT_VALUE + 0.1)

plt.title(f"Model confidence")

plt.draw()
plt.pause(1)

plt.savefig(f"plots/no_ood_contour.png")
