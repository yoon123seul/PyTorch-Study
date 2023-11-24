def sigmoidFn(self, x):
	return 1 / (1 + torch.exp(-x))
def ReLU(self, x):
	return torch.max(torch.tensor(0.).to(device), x)
def LeakyReLU(self, x):
	return torch.max(0.01*x, x)
