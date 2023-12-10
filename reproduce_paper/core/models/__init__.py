from .toy import toy, toy_maxy
from .doubleintegrator_nonres import doubleintegrator_nonres, doubleintegrator_nonres_ulimits, doubleintegrator_orig

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

