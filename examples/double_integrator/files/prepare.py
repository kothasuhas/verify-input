import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Dynamics(nn.Module):
    def __init__(self, A, B, c, policy, constraints, n_steps=1):
        super(Dynamics, self).__init__()

        self.policy = policy

        A = torch.tensor(A, dtype=torch.float32)
        B = torch.tensor(B, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        constraints = torch.tensor(constraints, dtype=torch.float32)

        input_dim = A.shape[1]
        c_count = constraints.shape[0]

        self.apply_c = nn.Linear(input_dim, input_dim + c_count, bias=True)
        self.apply_c.weight.data[:input_dim] = torch.eye(A.shape[1])
        self.apply_c.weight.data[input_dim:] = constraints
        self.apply_c.bias.data = torch.zeros((input_dim + c_count))

        self.remove_c = nn.Linear(input_dim + c_count, input_dim, bias=True)
        self.remove_c.weight.data[:, :input_dim] = torch.eye(A.shape[1])
        self.remove_c.weight.data[:, input_dim:] = torch.zeros((input_dim, c_count))
        self.remove_c.bias.data = torch.zeros((input_dim))
        
        self.A = nn.Linear(A.shape[1], A.shape[0], bias=False)
        self.A.weight.data = torch.tensor(A, dtype=torch.float32)

        self.B = nn.Linear(B.shape[1], B.shape[0], bias=False)
        self.B.weight.data = torch.tensor(B, dtype=torch.float32)

        self.c = c.T

        self.n_steps = n_steps

    def forward(self, x):
        x = self.apply_c(x)
        x = self.remove_c(x)
        x = nn.Flatten()(x)
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
double_integrator_policy.load_state_dict(torch.load('files/double_integrator.pt'))

num_cs = 20
cs = torch.tensor([[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)])
cs = torch.where(torch.abs(cs) > 0.0001, cs, 0.0001)

for i in range(1, 11):
    DoubleIntegrator = Dynamics(
        A=[[1.0, 1.0], [0.0, 1.0]],
        B=[[0.5], [1.0]],
        c=[[0.0], [0.0]],
        policy=double_integrator_policy,
        constraints=cs,
        n_steps=i,
    )
    DoubleIntegrator.eval()

    dummy_input = torch.randn(1, 2)

    torch.onnx.export(DoubleIntegrator, dummy_input, f"files/double_integrator_{i}.onnx")
