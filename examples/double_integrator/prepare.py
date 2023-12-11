import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Dynamics(nn.Module):
    def __init__(self, A, B, c, policy, cs, n_steps=1):
        super(Dynamics, self).__init__()

        self.policy = policy

        A = torch.tensor(A, dtype=torch.float32)
        B = torch.tensor(B, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        cs = torch.tensor(cs, dtype=torch.float32)

        input_dim = A.shape[1]
        c_count = cs.shape[0]

        self.apply_c = nn.Linear(input_dim, input_dim + c_count, bias=True)
        self.apply_c.weight.data[:input_dim] = torch.eye(A.shape[1])
        self.apply_c.weight.data[input_dim:] = cs
        self.apply_c.bias.data = torch.zeros((input_dim + c_count))

        self.remove_c = nn.Linear(input_dim + c_count, input_dim, bias=True)
        self.remove_c.weight.data[:, :input_dim] = torch.eye(A.shape[1])
        self.remove_c.weight.data[:, input_dim:] = torch.zeros((input_dim, c_count))
        self.remove_c.bias.data = torch.zeros((input_dim))
        
        self.A = nn.Linear(A.shape[1], A.shape[0], bias=True)
        self.A.weight.data = torch.tensor(A, dtype=torch.float32)
        self.A.bias.data = torch.zeros((A.shape[0]))

        self.B = nn.Linear(B.shape[1], B.shape[0], bias=True)
        self.B.weight.data = torch.tensor(B, dtype=torch.float32)
        self.B.bias.data = torch.zeros((B.shape[0]))

        self.c = c.T

        self.n_steps = n_steps

    def forward(self, x):
        x = self.apply_c(x)
        x = self.remove_c(x)
        for _ in range(self.n_steps):
            u = self.policy(x)
            x = self.A(x) + self.B(u) + self.c
        return x

def doubleintegrator_orig():
    STATE_DIM = 2
    HIDDEN1_DIM = 10
    HIDDEN2_DIM = 5
    POLICY_DIM = 1

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(STATE_DIM, HIDDEN1_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM, POLICY_DIM)
    )
    return model

double_integrator_policy = doubleintegrator_orig()
double_integrator_policy.load_state_dict(torch.load('double_integrator.pt'))

planes = [[1.0, 0.0], [0.9876883405951378, 0.15643446504023087], [0.9510565162951535, 0.3090169943749474], [0.8910065241883679, 0.45399049973954675], [0.8090169943749475, 0.5877852522924731], [0.7071067811865476, 0.7071067811865475], [0.5877852522924732, 0.8090169943749473], [0.4539904997395468, 0.8910065241883678], [0.30901699437494745, 0.9510565162951535], [0.15643446504023092, 0.9876883405951378], [6.123233995736766e-17, 1.0], [-0.1564344650402306, 0.9876883405951378], [-0.30901699437494734, 0.9510565162951536], [-0.45399049973954675, 0.8910065241883679], [-0.587785252292473, 0.8090169943749475], [-0.7071067811865475, 0.7071067811865476], [-0.8090169943749473, 0.5877852522924732], [-0.8910065241883678, 0.45399049973954686], [-0.9510565162951535, 0.3090169943749475], [-0.9876883405951377, 0.15643446504023098]]

for i in range(1, 11):
    DoubleIntegrator = Dynamics(
        A=[[1.0, 1.0], [0.0, 1.0]],
        B=[[0.5], [1.0]],
        c=[[0.0], [0.0]],
        policy=double_integrator_policy,
        cs=planes,
        n_steps=i,
    )
    DoubleIntegrator.eval()

    dummy_input = torch.randn(1, 2)

    torch.onnx.export(DoubleIntegrator, dummy_input, f"double_integrator_{i}.onnx")
