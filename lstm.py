import torch
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, mean_pooling, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(int(input_size), int(hidden_dim/2), num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()
        self.mean_pooling = mean_pooling
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, device=self.device),
                torch.randn(2, 1, self.hidden_dim // 2, device=self.device))

    def forward(self, X):
        self.hidden = self.init_hidden()
        steps = X.size()[0]
        lstm_out, self.hidden = self.lstm(X, self.hidden)
        lstm_out = lstm_out.view(steps, self.hidden_dim)

        if self.mean_pooling:
            print("Mean pooling over the output...")
            out = torch.mean(lstm_out)
        else:
            print("Using the last time step...")
            out = lstm_out[-1]

        out = F.leaky_relu(self.fc1(out), negative_slope=.2)
        out = F.leaky_relu(self.fc2(out), negative_slope=.2)
        out = self.fc3(out)

        return out
