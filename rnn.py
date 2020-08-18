import torch
import torch.nn.functional as F

class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_out, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers,
            batch_first=True, dropout=drop_out)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(self.n_layers, batch_size,
            self.hidden_dim).to(self.device)
        _, hidden = self.rnn(x, hidden)

        # This does some reshaping, might be an old idiom
        # see: https://discuss.pytorch.org/t/when-and-why-do-we-use-contiguous/47588
        hidden = hidden.contiguous().view(-1, self.hidden_dim)
        hidden = self.fc(hidden)

        return hidden
