import torch
import torch.nn.functional as F

class RowNet(torch.nn.Module):
    def __init__(self, input_size, embedded_dim=1024):
        # Language (BERT): 3072, Vision + Depth (ResNet152): 2048 * 2
        super(RowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, input_size)
        self.fc3 = torch.nn.Linear(input_size, embedded_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=.2)
        x = self.fc3(x)

        return x
