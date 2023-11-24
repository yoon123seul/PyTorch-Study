import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 생성되는 이미지 관찰
from matplotlib import cm # 데이터 포인트에 색상을 입힘

num_epochs = 50
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
    )
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
    )

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
    )
test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
    )

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Sequential을 사용해 encoder와 decoder 두 모듈로 묶어줌
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3), # 입력의 특징을 3차원으로 압축(출력값이 잠재변수)
            )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(), # 픽셀당 0~1 사이의 값을 가지도록 함
            )

    def forward(self, x):
        encoded = self.encoder(x) # encoder를 통해 입력을 잠재변수로 변환
        decoded = self.decoder(encoded) # decoder를 통해 잠재변수를 출력(복원 이미지)으로 변환
        return encoded, decoded # 잠재변수와 출력값을 반환

autoencoder = Autoencoder().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()

# 원본 이미지 시각화 하기
view_data = train_dataset.data[:10].view(-1, 28*28)
# 복원이 어떻게 되는지 관찰하기 위해 10개의 이미지를 가져와 바로 넣어봄
view_data = view_data.to(dtype=torch.float) / 255.

train_losses = []
test_losses = []

# train 함수 정의
def train(autoencoder, train_loader, optimizer, criterion):
    autoencoder.train()
    avg_loss = 0.0
    for step, (x, _) in enumerate(train_loader):  # label은 사용하지 않으므로 _로 대체
        x = x.view(-1, 28*28).to(device)
        y = x.view(-1, 28*28).to(device)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    avg_loss /= len(train_loader)
    return avg_loss

# test 함수 정의
def test(autoencoder, test_loader, criterion):
    autoencoder.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(-1, 28*28).to(device)
            y = x.view(-1, 28*28).to(device)

            encoded, decoded = autoencoder(x)
            test_loss += criterion(decoded, y).item()
    test_loss /= len(test_loader)
    return test_loss

# 잠재변수 시각화를 위한 데이터 로딩
# 여기서는 전체 트레이닝 데이터셋 대신 처음 200개의 데이터 포인트만 사용
view_data = train_dataset.data[:200].view(-1, 28*28).type(torch.float) / 255.0
view_data = view_data.to(device)
classes = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
    }

autoencoder.eval()
with torch.no_grad():
    encoded_data, _ = autoencoder(view_data)
encoded_data = encoded_data.to('cpu').detach().numpy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

X = encoded_data[:, 0]
Y = encoded_data[:, 1]
Z = encoded_data[:, 2]

labels = train_dataset.targets[:200].numpy()

for x, y, z, s in zip(X, Y, Z, labels):
    name = classes[s]
    color = cm.rainbow(int(255 * s / 9))  # 레이블에 따라 색깔 지정
    ax.text(x, y, z, name, color=color)  # 레이블을 잠재 벡터 위치에 텍스트로 표시

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
