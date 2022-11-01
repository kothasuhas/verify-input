from .utils import seed

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, args):
        super(Trainer, self).__init__()
        
        seed(1)
        self.model = create_model(args.model, device)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.init_optimizer(self.args.num_epochs)

    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                         weight_decay=5e-4, momentum=0.9, nesterov=True)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, 
                                                             pct_start=0.025, total_steps=int(num_epochs))
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)

            loss, batch_metrics = self.standard_loss(x, y)
            loss.backward()
            self.optimizer.step()
            
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        self.scheduler.step()
        return dict(metrics.mean())
    
    def accuracy(self, true, preds):
        """
        Computes multi-class accuracy.
        Arguments:
            true (torch.Tensor): true labels.
            preds (torch.Tensor): predicted labels.
        Returns:
            Multi-class accuracy.
        """
        accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true.argmax(dim=1)).sum().float()/float(true.size(0))
        return accuracy.item()
    

    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        out = torch.softmax(self.model(x),dim=1)
        loss = self.criterion(out, y)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': self.accuracy(y, preds)}
        return loss, batch_metrics
    
    
    def eval(self, dataloader):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = self.model(x)
            acc += self.accuracy(y, out)
        acc /= len(dataloader)
        return acc

    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        weights_path = 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
            # raise RuntimeError('Model weights not found at {}.'.format(path))
        self.model.load_state_dict(checkpoint)