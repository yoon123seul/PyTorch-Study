class LinearClassificationModel(nn.Module):
    def __init__(self, activation_fn):
        super(LinearClassificationModel, self).__init__()

        self.linear1 = nn.Linear(in_features = 32*32*3, out_features = 4096)
        self.linear2 = nn.Linear(in_features = 4096, out_features = 1024)
        self.linear3 = nn.Linear(in_features = 1024, out_features = 256)
        self.linear4 = nn.Linear(in_features = 256, out_features = 10)

        if activation_fn == 'sigmoid':
            self.activation = self.sigmoidFn
        elif activation_fn == 'ReLU':
            self.activation = self.ReLU
        elif activation_fn == 'LeakyReLU':
            self.activation = self.LeakyReLU
    
    def sigmoidFn(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def ReLU(self, x):
        return torch.max(0, x)

    def LeakyReLU(self, x):
        return torch.max(0.01*x, x)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x

optimizer = torch.optim.Adam(LCM.parameters(), lr = learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.CrossEntropyLoss().to(device)
