import torch

from .convnet import convnet
from .mlp import mlp
from .toy import toy
from .vae import vae
from .doubleintegrator_nonres import doubleintegrator_nonres, doubleintegrator_nonres_ulimits

def create_model(name, device):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    return eval(name)().to(device)

