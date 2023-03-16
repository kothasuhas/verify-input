from .toy import toy, toy_maxy
from .doubleintegrator_nonres import doubleintegrator_nonres, doubleintegrator_nonres_ulimits, doubleintegrator_orig
from .srgr_ffns import ffn2, ffn3, ffn4, ffn5, ffn6, ffn7, ffn8

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

