class Model(nn.Module): 
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32*32*3, 4096)
        self.linear2 = nn.Linear(4096, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 10)
 
    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)

        return x
