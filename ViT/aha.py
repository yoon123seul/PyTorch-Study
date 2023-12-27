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

BATCH_SIZE=param['BATCH_SIZE']
P=param['Patch']
COLOR_SIZE=param['Color']
IMG_SIZE=224
EMB_SIZE=COLOR_SIZE*P*P
NUM_HEADS=param['Num_head']
DROP_OUT=param['Drop_out']
# embed x shape : torch.Size([8, 197, 768])

x=torch.randn(NUM_HEADS, 197, EMB_SIZE)
class MHA(nn.Module):
    def __init__(self, emb_size:int=EMB_SIZE, num_heads:int=NUM_HEADS, dropout:float=DROP_OUT):
        super().__init__()
        self.emb_size=emb_size
        self.num_heads=num_heads
        self.head_emb = emb_size//num_heads
        self.q=nn.Linear(emb_size, emb_size)
        self.k=nn.Linear(emb_size, emb_size)
        self.v=nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x):
        Q=self.q(x)
        K=self.k(x)
        V=self.v(x)
        Q=Q.view(BATCH_SIZE, self.num_heads, -1, self.head_emb)
        K=K.view(BATCH_SIZE, self.num_heads, -1, self.head_emb)
        V=V.view(BATCH_SIZE, self.num_heads, -1, self.head_emb)
        
        energy=torch.matmul(Q, K.permute(0, 1, 3, 2))
        scale = self.emb_size**(1/2)
        att = F.softmax(energy, dim=-1) / scale
        out = torch.matmul(att, V)
        out = out.view(BATCH_SIZE, -1, self.num_heads*self.head_emb)
        out = self.projection(out)
        return out
    

# embed=MHA()
# x=embed(x)
# print("mha x shape :", x.shape)
