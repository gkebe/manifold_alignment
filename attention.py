import torch
from torch import nn

class Flatten2(nn.Module):
    """
    Takes a vector of shape (A, B, C, D, E, ...)
    and flattens everything but the first two dimensions,
    giving a result of shape (A, B, C*D*E*...)
    """
    def forward(self, input):
        return input.view(input.size(0), input.size(1), -1)

class Combiner(nn.Module):
    """
    This class is used to combine a feature exraction network F and a importance prediction network W,
    and combine their outputs by adding and summing them together.
    """
    def __init__(self, D, neurons):
        """
        featureExtraction: a network that takes an input of shape (B, T, D) and outputs a new
            representation of shape (B, T, D').
        weightSelection: a network that takes in an input of shape (B, T, D) and outputs a
            tensor of shape (B, T, 1) or (B, T). It should be normalized, so that the T
            values at the end sum to one (torch.sum(_, dim=1) = 1.0)
        """
        print("Using attention poling")
        super(Combiner, self).__init__()
        self.featureExtraction = nn.Sequential(
            Flatten2(),  # Shape is now (B, T, D)
            nn.Linear(D, neurons),  # Shape becomes (B, T, neurons)
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
        )
        self.weightSelection = nn.Sequential(
            # Shape is (B, T, neurons)
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, 1),  # (B, T, 1)
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        """
        input: a tensor of shape (B, T, D)
        return: a new tensor of shape (B, D')
        """
        print(input.shape)
        features = self.featureExtraction(input)  # (B, T, D)
        # weights = self.weightSelection(input) #(B, T) or (B, T, 1)
        weights = self.weightSelection(features)  # (B, T) or (B, T, 1)
        if len(weights.shape) == 2:  # (B, T) shape
            weights.unsqueese(2)  # now (B, T, 1) shape

        r = features * weights  # (B, T, D) shape

        return torch.sum(r, dim=1)  # sum over the T dimension, giving (B, D) final shape