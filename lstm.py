import torch
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, dropout, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size, hidden_dim, num_layers,
            batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(self.num_layers, batch_size,
            self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.num_layers, batch_size,
            self.hidden_dim).to(self.device)
        output, (hidden, c_n) = self.lstm(x, (hidden, c_0))

        # This does some reshaping, might be an old idiom
        # see: https://discuss.pytorch.org/t/when-and-why-do-we-use-contiguous/47588
        #hidden = self.contiguous().view(-1, self.hidden_dim)
        hidden = self.fc(hidden)

        return hidden
