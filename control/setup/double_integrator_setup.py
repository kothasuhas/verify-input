# code to convert a keras controls policy into a torch nonresidual network
# (no need to run unless you have a new controls policy to test)

import torch
from torch import nn

from keras.models import model_from_json

from core.models.doubleintegrator_nonres import doubleintegrator_orig, doubleintegrator_nonres, doubleintegrator_nonres_ulimits, STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM
from .controls_construct_fullstep import construct_full_step

# assert Flatten, Linear, Relu, Linear, Relu, Linear

u_limits = False
if u_limits:
    u_lb = torch.Tensor([-1.0])
    u_ub = torch.Tensor([1.0])
else:
    u_lb, u_ub = None, None

with open('control/models/doubleintegrator.json', 'r') as json_file:
    keras_model_json = json_file.read()
keras_model = model_from_json(keras_model_json)
keras_model.load_weights("control/models/doubleintegrator.h5")

orig_model = doubleintegrator_orig()
if u_limits:
    full_step = doubleintegrator_nonres_ulimits()
else:
    full_step = doubleintegrator_nonres()

A = torch.Tensor([[1.0, 1.0], [0.0, 1.0]])
B = torch.Tensor([[0.5], [1.0]])

orig_model, full_step = construct_full_step(keras_model, orig_model, full_step, 
                                            u_limits, u_lb, u_ub, 
                                            A, B,
                                            STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM)

torch.save(orig_model.state_dict(), 'control/models/doubleintegrator_orig.pt')
if u_limits:
    torch.save(full_step.state_dict(), 'control/models/doubleintegrator_ulimits1.pt')
else:
    torch.save(full_step.state_dict(), 'control/models/doubleintegrator.pt')
