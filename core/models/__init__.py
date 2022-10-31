import torch

from .convnet import convnet
from .mlp import mlp
from .i_featurizer import featurizer
from .toy import toy
from .vae import vae

def create_model(name, device):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'convnet':
        model = convnet()
    elif name == 'mlp':
        model = mlp()
    elif name == 'featurizer':
        model = featurizer()
    elif name == 'toy':
        model = toy()
    elif name == 'vae':
        model = vae()
    else:
        raise ValueError('Invalid model name {}!'.format(name))

    model = model.to(device)
    return model

