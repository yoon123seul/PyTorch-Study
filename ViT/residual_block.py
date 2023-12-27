import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchsummary import summary
import yaml

with open('param.yml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

EXP=param['expansion']
DROP_OUT=param['Drop_out']

class ResidualAdd(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x+= res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self,emb_size:int, expansion : int=EXP, drop_p :float = DROP_OUT):
        super().__init__(
            nn.Linear(emb_size, expansion* emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion*emb_size, emb_size))
