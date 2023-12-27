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

x=torch.randn(BATCH_SIZE, COLOR_SIZE, IMG_SIZE, IMG_SIZE)

class PatchEmbedding(nn.Module):
    def __init__(self,
                 color_channel:int=COLOR_SIZE,
                 patch_size:int=P,
                 emb_size:int=EMB_SIZE,
                 image_size:int=IMG_SIZE):
        super().__init__()
        self.projection=nn.Sequential(
            nn.Conv2d(COLOR_SIZE, EMB_SIZE, kernel_size=P, stride=P)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMB_SIZE))
        self.position = nn.Parameter(torch.randn((IMG_SIZE//P)**2+1, EMB_SIZE))
    def forward(self, x):
        x=self.projection(x)
        x=x.view(BATCH_SIZE, -1, EMB_SIZE)
        # print(x.shape)
        rp_token=self.cls_token.repeat(BATCH_SIZE, 1, 1)
        x=torch.cat([rp_token, x], dim=1)
        # print(x.shape)
        x+=self.position
        return x
    
# conv_projection=nn.Sequential(
#     nn.Conv2d(COLOR_SIZE, EMB_SIZE, kernel_size=P, stride=P)
# )
# x=conv_projection(x)
# patched=x.view(BATCH_SIZE, -1, EMB_SIZE)
# # print("Projected X shape : ", patched.shape)

# token = nn.Parameter(torch.zeros(1, 1, EMB_SIZE))
# position = nn.Parameter(torch.randn((IMG_SIZE//P)**2+1, EMB_SIZE))
# # print("token shape : ", token.shape)
# # print("Position shape : ", position.shape)

# rp_token = token.repeat(BATCH_SIZE, 1, 1)
# # print(rp_token.shape)
# catx = torch.cat([rp_token, patched], dim=1)
# print("classtoken_added:", catx.shape)
# catx += position
# print("Result:", catx.shape)


# print("x shape :", x.shape)
# embed=PatchEmbedding()
# x=embed(x)
# print("embed x shape :", x.shape)
