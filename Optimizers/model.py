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
optimzers = ["SGD", "Adam", "AdamW"]
result = {}
for opt in optimzers:
    train_loss_opt = []
    test_accuracy_opt = []
    if opt == "SGD":
            optimizer = torch.optim.SGD(TestModel.parameters(), lr=learning_rate)
    elif opt == "Adam":
            optimizer = torch.optim.Adam(TestModel.parameters(), lr=learning_rate)
    elif opt == "AdamW":
            optimizer = torch.optim.AdamW(TestModel.parameters(), lr=learning_rate)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) ##########

    for epoch in range(nb_epochs):
        train_loss = train(TestModel, trainloader, optimizer, criterion)
        test_accuracy = test(TestModel, testloader, criterion)
        scheduler.step()
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        train_loss_opt.append(train_loss)
        test_accuracy_opt.append(test_accuracy)

    result[opt] = {"Train_loss" : train_loss_opt, "Accuracy" : test_accuracy_opt}

print (result)
