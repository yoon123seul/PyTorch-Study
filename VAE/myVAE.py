import os
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import save_image
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("사용하는 Device : ", device)

image_path = './images'
n_epochs = 100
batch_size = 128
lr = 1e-3
b1 = 0.5
b2 = 0.999
img_size = 28
hidden_dim = 400
latent_dim = 10

os.makedirs(image_path, exist_ok=True)

transform = transforms.Compose([
            transforms.ToTensor()
            ])

train = datasets.MNIST(root='./data/',train=True,transform=transform,download=True)
test = datasets.MNIST(root='./data/',train=False,transform=transform,download=True)

train_dataloader = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,

)

test_dataloader = torch.utils.data.DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
)

def reparameterization(mu, std):
    eps = torch.randn_like(std)
    return mu + eps * std

class Encoder(nn.Module):
    def __init__(self, x_dim=img_size**2, h_dim=hidden_dim, z_dim=latent_dim):
        super(Encoder, self).__init__()

        # 2 hidden layer
        self.input_layer = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )


        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.input_layer(x)
        output_layer = nn.Sigmoid()
        mu = output_layer(self.mu(x))
        std = output_layer(self.std(x))



        z = reparameterization(mu, std)
        return z, mu, std
    
class Decoder(nn.Module):
    def __init__(self, x_dim=img_size**2, h_dim=hidden_dim, z_dim=latent_dim):
        super(Decoder, self).__init__()

        # 2 hidden layer + outpur layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_reconst = self.fc1(z)
        return x_reconst
        

    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)

print(encoder)
print(decoder)


train_loss_list = []
test_loss_list = []
for epoch in range(n_epochs):
    train_loss = 0
    for (x, _) in tqdm (train_dataloader):
        # forward
        x = x.view(-1, img_size**2)
        x = x.to(device)
        z, mu, std = encoder(x)
        x_reconst = decoder(z)

        reconst_loss = nn.functional.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - (std.pow(2).log()) - 1)
        # print ("kl_div", kl_div)

        # backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'===> Epoch: {epoch+1} Average Train Loss: {train_loss/len(train_dataloader.dataset):.4f} ')
    train_loss_list.append (train_loss)

    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(test_dataloader)):
            # forward
            x = x.view(-1, img_size**2)
            x = x.to(device)
            z, mu, std = encoder(x)
            x_reconst = decoder(z)

            reconst_loss = nn.functional.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - (std.pow(2).log()) - 1)

            loss = reconst_loss + kl_div
            test_loss += loss.item()

            if i==0:
                x_concat = torch.cat([x.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, os.path.join(image_path,f'reconst-epoch{epoch+1}.png'))

        print(f'===> Epoch: {epoch+1} Average Test Loss: {test_loss/len(test_dataloader.dataset):.4f} ')
        test_loss_list.append(test_loss)



import matplotlib.pyplot as plt
x = [ i + 1 for i in range (100) ]
# print(x)
y= [y / 60000 for y in train_loss_list]
plt.plot(x, y, linestyle='-')  # 선 그래프
plt.title('train loss')  # 그래프 제목
plt.xlabel('epoch')  # x 축 레이블
plt.ylabel('train loss')  # y 축 레이블
plt.show()


x = [ i + 1 for i in range (100) ]
# print(x)
y= [y / 10000 for y in test_loss_list]
plt.plot(x, y, linestyle='-')  # 선 그래프
plt.title('test loss')  # 그래프 제목
plt.xlabel('epoch')  # x 축 레이블
plt.ylabel('test loss')  # y 축 레이블
plt.show()
