import torch
import numpy as np
from crown.driver import optimize
from crown.model_utils import load_model
from crown.plot_utils import PlottingLevel

models = [load_model(f"ffn{i}", f"clean/ffn{i}.pt") for i in range(2, 9)]