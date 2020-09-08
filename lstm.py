import torch
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, awe, num_layers, dropout, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.awe = awe
        self.num_layers = num_layers

        self.k = None

        self.lstm = torch.nn.LSTM(input_size, hidden_dim, num_layers,
            batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim * awe, output_size)

    def set_TBPTT(self, h):
        '''
        Sets k1 = k2 = h for Truncated Back Propogation through time
        to avoid exploding gradients and speed up training time for
        long sequences.

        Williams, RJ and Peng, J "An efficient gradient-based algorithm for on-line
        training of recurrent network trajectories". 1990
        '''
        self.k = h
        
        return self

    def forward(self, X):
        batch_size = X.size(0)
        h_t = torch.zeros(self.num_layers, batch_size,
            self.hidden_dim).to(self.device)
        c_t = torch.zeros(self.num_layers, batch_size,
            self.hidden_dim).to(self.device)

        if self.k is not None:
            T = X.size(1)
            for t, x in enumerate(X.split(1, 1)):
                #print(x)
                #print(X.size())
                #print(x.size())

                out, (h_t, c_t) = self.lstm(x, (h_t, c_t))
                # perform TBPTT if set
                if T - t == self.k:
                    out.detach()
        else:
            out, (h_t, c_t) = self.lstm(X, (h_t, c_t))
            seq_len = out.size(1)
            #print(f'init size: {out.size()}')
            out = out.narrow(
                1,
                max(0, seq_len - self.awe),
                min(self.awe, seq_len)
            ).flatten(1)
            print(len(out))
            #print(f'out  size: {out.size()}')

        #hidden = h_t.view(self.num_layers, batch_size, self.hidden_dim)[-1]
        #print(f'hidden size: {hidden.size()}')
        # This does some reshaping, might be an old idiom
        # see: https://discuss.pytorch.org/t/when-and-why-do-we-use-contiguous/47588
        #hidden = self.contiguous().view(-1, self.hidden_dim)
        #hidden = self.fc(hidden)

        out = self.fc(out)

        return out
