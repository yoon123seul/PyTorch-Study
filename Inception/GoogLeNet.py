class GoogLeNet(nn.Module):

    def __init__(
            self,
            num_classes=10,
            aux_logits=True
            ):
        super().__init__()

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3) # input channels, output channels,kernel_size,stride,padding
        self.maxpool1 = nn.MaxPool2d(3, 2, 1) #kernel, stride, padding

        self.conv2a = conv_block(64, 64, kernel_size=1)
        self.conv2b = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)

        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32) #input_channels, n_1x1, n_3x3red, n_3x3, n_5x5_red, n_5x5, pool_proj
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2, 1)
        
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.aux_logits = aux_logits    
        self.avgpool = nn.AvgPool2d(7, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = Inception_aux(512, num_classes)
            self.aux2 = Inception_aux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
            aux1=F.softmax(aux1,dim=1)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
            aux2=F.softmax(aux2,dim=1)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x
          
class conv_block(nn.Module):
  def __init__(self, input_channels, output_channels, **kwargs):
      super().__init__()

      self.conv = nn.Sequential(
          nn.Conv2d(input_channels, output_channels, **kwargs),
          nn.BatchNorm2d(output_channels),
          nn.ReLU(),
      )
  
  def forward(self, x):
      return self.conv(x)

class Inception_block(nn.Module):
  def __init__(self, input_channels, n_1x1, n_3x3_red, n_3x3, n_5x5_red, n_5x5, poll_proj):
      super().__init__()
      self.branch1 = conv_block(input_channels, n_1x1, kernel_size=1)
      self.branch2 = nn.Sequential(
          conv_block(input_channels, n_3x3_red, kernel_size=1),
          conv_block(n_3x3_red, n_3x3, kernel_size=3, padding=1),
      )

      self.branch3 = nn.Sequential(
          conv_block(input_channels, n_5x5_red, kernel_size=1),
          conv_block(n_5x5_red, n_5x5, kernel_size=5, padding=2),
      )

      self.branch4 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
          conv_block(input_channels, poll_proj, kernel_size=1)
      )

  def forward(self, x):
      # filter 수 dim=1 
      x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
      return x
    
  
class Inception_aux(nn.Module):
  def __init__(self, in_channels, num_classes):
      super().__init__()

      self.conv = nn.Sequential(
          nn.AvgPool2d(kernel_size=5, stride=3),
          conv_block(in_channels, 128, kernel_size=1),
      )

      self.fc = nn.Sequential(
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(),#p=0.5
          nn.Linear(1024, num_classes),
      )

  def forward(self,x):
      x = self.conv(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
      return x


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
