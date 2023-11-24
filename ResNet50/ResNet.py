class BottleNeck(nn.Module):
    mul = 4
    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.conv3 = nn.Conv2d(out_planes, out_planes*self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes*self.mul)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != out_planes*self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes*self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes*self.mul)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


 class ResNet(nn.Module):
		def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.linear = nn.Linear(512 * block.mul, num_classes)
  
    def make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1) # 첫번째만 stride=2(사이즈 절반으로 줄임), 나머지는 1
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out


def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

model = ResNet50().to(device)

transform_d=transforms.Compose([
    v2.RandAugment(num_ops=2,magnitude=5),
    transforms.ToTensor(),
    transforms.Resize(224,antialias=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set=datasets.STL10('./stl-10',split='train',download=True,transform=transform_d)
test_set=datasets.STL10('./stl-10',split='test',download=True,transform=transform_d)

minibatch_size=128
train_batches = DataLoader(train_set, batch_size=minibatch_size, shuffle=True)
test_batches = DataLoader(test_set, batch_size=minibatch_size, shuffle=False)


def train_model(model):
    model.train() # prep model for training
    train_loss=0
    for x_minibatch, y_minibatch in train_batches:
        x_minibatch = x_minibatch.to(device)
        y_minibatch = y_minibatch.to(device)              
        y_minibatch_pred = model(x_minibatch)
        if len(y_minibatch_pred) == 3:
            y_minibatch_pred, aux1, aux2 = y_minibatch_pred
            aux1_loss = loss_func(aux1, y_minibatch)
            aux2_loss = loss_func(aux2, y_minibatch)
            loss = loss_func(y_minibatch_pred, y_minibatch) + 0.3 * (aux1_loss + aux2_loss)
        else:
            loss = loss_func(y_minibatch_pred, y_minibatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_batches)
    print('Average Train Loss: {:.4f}'.format(train_loss))


def test_model(model):
    test_loss = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for x_minibatch, y_minibatch in test_batches:
            x_minibatch = x_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)      
            y_test_pred = model(x_minibatch)
            test_loss += loss_func(y_test_pred, y_minibatch)  
            pred = torch.argmax(y_test_pred, dim=1)
            correct += pred.eq(y_minibatch).sum().item()


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = GoogLeNet(aux_logits=True, num_classes=10).to(device)

loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)

max_epoch=50
for epoch in range(max_epoch):
    print(f'Epoch{epoch+1}')
    train_model(model)
    scheduler.step()
test_model(model)
