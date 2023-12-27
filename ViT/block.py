import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch import nn
from torch import Tensor
from PIL import Image
from torchsummary import summary
from mha import MHA
from residual_block import ResidualAdd, FeedForwardBlock
from patchembedding import PatchEmbedding

with open('param.yml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

BATCH_SIZE=param['BATCH_SIZE']
P=param['Patch']
COLOR_SIZE=param['Color']
IMG_SIZE=224
EMB_SIZE=COLOR_SIZE*P*P
NUM_HEADS=param['Num_head']
DROP_OUT=param['Drop_out']
EXP=param['expansion']
DEPTH=param['Depth']

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                emb_size:int=EMB_SIZE,
                drop_p : float=DROP_OUT,
                forward_expansion : int=EXP,
                forward_drop_p : float=DROP_OUT,
                **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MHA(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion,
                    drop_p = forward_drop_p),
                )
            ))
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth:int=DEPTH, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size:int =EMB_SIZE, n_classes: int=1000):
        super().__init__()
        self.layernorm=nn.LayerNorm(emb_size)
        self.fc=nn.Linear(emb_size, n_classes)
        
    def forward(self, x):
        x=torch.mean(x, 1)
        x=self.layernorm(x)
        x=self.fc(x)
        return x
        
class ViT(nn.Sequential):
    def __init__(self, 
                in_channels: int=COLOR_SIZE,
                patch_size : int=P,
                emb_size :int=EMB_SIZE,
                img_size:int=224,
                depth: int=DEPTH,
                n_classes:int =1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
        TransformerEncoder(depth, emb_size=emb_size, **kwargs),
        ClassificationHead(emb_size, n_classes))
