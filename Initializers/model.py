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

initializers = ["zeros", "gaussian", "xavier", "he"]
result_init = {}
for init in initializers:
    
    train_loss_init = []
    test_accuracy_init = []

    
    TestModel = Model(initializer = init).to(device)

    learning_rate = 0.001
    weight_decay = 0.1
    optimizer = torch.optim.AdamW(TestModel.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) ##########
    criterion = nn.CrossEntropyLoss()
    nb_epochs = 30

    for epoch in range(nb_epochs):
        train_loss_init = train(TestModel, trainloader, optimizer, criterion)
        test_accuracy_init = test(TestModel, testloader)
        scheduler.step()
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    result_init[init] = {"Train_loss" : train_loss_init, "Accuracy" : test_accuracy_init}

print (result_init)
