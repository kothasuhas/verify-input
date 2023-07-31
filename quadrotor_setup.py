# code to convert a keras controls policy into a torch nonresidual network
# (no need to run unless you have a new controls policy to test)

import torch

from keras.models import model_from_json

from core.models.quadrotor import quadrotor_orig, quadrotor_nonres, quadrotor_nonres_ulimits, STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM
from controls_construct_fullstep import construct_full_step

# assert Flatten, Linear, Relu, Linear, Relu, Linear

u_limits = False
if u_limits:
    u_lb = torch.Tensor([-8.0, -8.0, -1.8])
    u_ub = torch.Tensor([8.0, 8.0, 17.8])
else:
    u_lb, u_ub = None, None

with open('quadrotor.json', 'r') as json_file:
    keras_model_json = json_file.read()
keras_model = model_from_json(keras_model_json)
keras_model.load_weights("quadrotor.h5")

orig_model = quadrotor_orig()
if u_limits:
    full_step = quadrotor_nonres_ulimits()
else:
    full_step = quadrotor_nonres()

A = torch.zeros((6, 6))
A[0][3] = 1
A[1][4] = 1
A[2][5] = 1
B = torch.zeros((6, 3))
B[3][0] = 1
B[4][1] = -1
B[5][2] = 1

full_step = construct_full_step(keras_model, orig_model, full_step, 
                                u_limits, u_lb, u_ub, 
                                A, B,
                                STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM)
if u_limits:
    torch.save(full_step.state_dict(), 'quadrotor_ulimits_test.pt')
else:
    torch.save(full_step.state_dict(), 'quadrotor_test.pt')
