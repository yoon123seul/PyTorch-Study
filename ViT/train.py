import torch
import torch.nn as nn
import numpy as np
import os
from copy import deepcopy 
from torchvision import transforms,datasets
from tqdm import tqdm
import time
import wandb
import yaml

from mha import MHA
from residual_block import ResidualAdd, FeedForwardBlock
from patchembedding import PatchEmbedding
from block import TransformerEncoderBlock,TransformerEncoder,ClassificationHead,ViT
from torchvision.transforms import v2
from torch.utils.data import DataLoader

with open('param.yml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)


BATCH_SIZE=param['BATCH_SIZE']
PATCH=param['Patch']
COLOR_SIZE=param['Color']
IMG_SIZE=224
EMB_SIZE=COLOR_SIZE*PATCH*PATCH
NUM_HEADS=param['Num_head']
DROP_OUT=param['Drop_out']
DEPTH=param['Depth']
EPOCH=param['Epoch']

wandb.init(
    project="pytorchgod"
)

transform_d = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        v2.RandAugment(num_ops=2,magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

}

minibatch_size = BATCH_SIZE
train_set=datasets.ImageFolder('ImageNet1k-train',transform_d['train'])
val_set=datasets.ImageFolder('ImageNet1k-val',transform_d['val'])
# test_set=datasets.ImageFolder('ImageNet1k-test',transform_d['test'])

train_batches = DataLoader(train_set, batch_size=minibatch_size, shuffle=True,drop_last=True)
val_batches = DataLoader(val_set, batch_size=minibatch_size, shuffle=False,drop_last=True)
# test_batches = DataLoader(test_set, batch_size=minibatch_size, shuffle=False)

def train_model(model, n_epochs, progress_interval,loss_func,optimizer,device):
    
    train_losses, valid_losses, lowest_loss = list(), list(), np.inf

    for epoch in range(n_epochs):
        
        train_loss, valid_loss = 0, 0
        
        # train the model
        model.train() # prep model for training
        for x_minibatch, y_minibatch in tqdm(train_batches,desc="train...",position=0,mininterval=1.0):
            x_minibatch = x_minibatch.to(device)
            y_minibatch = y_minibatch.to(device) 
            y_minibatch_pred = model(x_minibatch)
            loss = loss_func(y_minibatch_pred, y_minibatch)

            optimizer.zero_grad()
            loss.backward()
            wandb.log({
                "step_loss": loss,
                "epoch": epoch,
            })
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_batches)
