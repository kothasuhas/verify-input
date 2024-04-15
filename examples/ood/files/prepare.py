import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

INPUT_DIM = 2
HIDDEN1_DIM = 200
HIDDEN2_DIM = 200
OUTPUT_DIM = 3

class ModifiedOod(nn.Module):
    def __init__(self, orig_model, cs):
        super(ModifiedOod, self).__init__()

        self.orig_model = orig_model

        num_cs = len(cs)
        cs = torch.tensor(cs, dtype=torch.float32)

        input_dim = INPUT_DIM
        c_count = cs.shape[0]

        self.apply_c = nn.Linear(input_dim, input_dim + c_count, bias=True)
        self.apply_c.weight.data[:input_dim] = torch.eye(input_dim)
        self.apply_c.weight.data[input_dim:] = cs
        self.apply_c.bias.data = torch.zeros((input_dim + c_count))

        self.remove_c = nn.Linear(input_dim + c_count, input_dim, bias=True)
        self.remove_c.weight.data[:, :input_dim] = torch.eye(input_dim)
        self.remove_c.weight.data[:, input_dim:] = torch.zeros((input_dim, c_count))
        self.remove_c.bias.data = torch.zeros((input_dim))

    def forward(self, x):
        x = self.apply_c(x)
        x = self.remove_c(x)
        return self.orig_model(x)

def ood_orig():

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_DIM, HIDDEN1_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM, OUTPUT_DIM)
    )
    return model

ood = ood_orig()
ood.load_state_dict(torch.load('files/ood.pt')['model_state_dict'])

# Note that the same cs are defined in the oc.py script!
num_cs = 20
cs = torch.tensor([[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)])
cs = torch.where(torch.abs(cs) > 0.0001, cs, 0.0001)

for i in range(1, 11):
    modified_ood = ModifiedOod(
        orig_model=ood,
        cs=cs,
    )
    modified_ood.eval()

    dummy_input = torch.randn(1, 2)

    torch.onnx.export(modified_ood, dummy_input, f"files/modified_ood.onnx")
