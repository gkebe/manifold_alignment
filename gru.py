import torch
import torch.nn.functional as F

class GRU(torch.nn.Module):
    def __init__(self, input_size, embedded_dim=1024):
        # Language (BERT): 3072, Vision + Depth (ResNet152): 2048 * 2
        super(GRU, self).__init__()
        self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=input_size, num_layers=1, dropout=0.3,
                          batch_first=True, bidirectional=False)
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, embedded_dim)

    def forward(self, x):
        _, h_n = self.rnn(x)
        x = h_n[-1, :, :]
        x = F.leaky_relu(self.fc1(x), negative_slope=.2)
        x = self.fc2(x)

        return x
