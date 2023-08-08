# code to convert a keras controls policy into a torch nonresidual network
# (no need to run unless you have a new controls policy to test)

# model taken from https://github.com/neu-autonomy/nfl_veripy/tree/release/src/nfl_veripy/_static/models/DiscreteQuadrotor/discrete_quad_avoid_origin_maneuver_2
# dynamics taken from https://github.com/neu-autonomy/nfl_veripy/blob/release/src/nfl_veripy/dynamics/DiscreteQuadrotor.py

import torch

from keras.models import model_from_json

from core.models.quadrotor import quadrotor_orig, quadrotor_nonres, quadrotor_nonres_ulimits, STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM
from .controls_construct_fullstep import construct_full_step

# assert Flatten, Linear, Relu, Linear, Relu, Linear

u_limits = True
if u_limits:
    u_lb = torch.Tensor([-4.0, -4.0, -4.0])
    u_ub = torch.Tensor([4.0, 4.0, 4.0])
else:
    u_lb, u_ub = None, None

with open('control/models/quadrotor.json', 'r') as json_file:
    keras_model_json = json_file.read()
keras_model = model_from_json(keras_model_json)
keras_model.load_weights("control/models/quadrotor.h5")

orig_model = quadrotor_orig()
if u_limits:
    full_step = quadrotor_nonres_ulimits()
else:
    full_step = quadrotor_nonres()

A = torch.Tensor(
    [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

B = torch.Tensor(
    [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

orig_model, full_step = construct_full_step(keras_model, orig_model, full_step, 
                                            u_limits, u_lb, u_ub, 
                                            A, B,
                                            STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM)
torch.save(orig_model.state_dict(), 'control/models/quadrotor_orig.pt')
if u_limits:
    torch.save(full_step.state_dict(), 'control/models/quadrotor_ulimits.pt')
else:
    torch.save(full_step.state_dict(), 'control/models/quadrotor.pt')
