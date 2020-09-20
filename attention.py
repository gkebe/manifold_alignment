import torch
from torch import nn
import torch.nn.functional as F
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


class AdditiveAttentionScore(nn.Module):

    def __init__(self, H):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(H, 1)
        self.w = nn.Linear(2 * H, H)  # 2*H because we are going to concatenate two inputs

    def forward(self, states, context):
        """
        states: (B, T, H) shape
        context: (B, H) shape
        output: (B, T, 1), giving a score to each of the T items based on the context

        """
        T = states.size(1)
        # Repeating the values T times
        context = torch.stack([context for _ in range(T)], dim=1)  # (B, D) -> (B, T, D)
        state_context_combined = torch.cat((states, context), dim=2)  # (B, T, D) + (B, T, D)  -> (B, T, 2*D)
        scores = self.v(torch.tanh(self.w(state_context_combined)))

        return scores
class ApplyAttention(nn.Module):
    """
    This helper module is used to apply the results of an attention mechanism to
    a set of inputs.
    """

    def __init__(self):
        super(ApplyAttention, self).__init__()

    def forward(self, states, attention_scores, mask=None):
        """
        states: (B, T, H) shape giving the T different possible inputs
        attention_scores: (B, T, 1) score for each item at each context
        mask: None if all items are present. Else a boolean tensor of shape
            (B, T), with `True` indicating which items are present / valid.

        returns: a tuple with two tensors. The first tensor is the final context
        from applying the attention to the states (B, H) shape. The second tensor
        is the weights for each state with shape (B, T, 1).
        """

        if mask is not None:
            # set everything not present to a score of -inf
            # gurantees it will not be seleced
            attention_scores[~mask] = float('-inf')
        # compute the weight for each score
        weights = F.softmax(attention_scores, dim=1)  # (B, T, 1) still, but sum(T) = 1

        final_context = (states * weights).sum(dim=1)  # (B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights


def getMaskByFill(x, time_dimension=1, fill=0):
    """
    x: the original input with three or more dimensions, (B, ..., T, ...)
        which may have unsued items in the tensor. B is the batch size,
        and T is the time dimension.
    time_dimension: the axis in the tensor `x` that denotes the time dimension
    fill: the constant used to denote that an item in the tensor is not in use,
        and should be masked out (`False` in the mask).

    return: A boolean tensor of shape (B, T), where `True` indicates the value
        at that time is good to use, and `False` that it is not.
    """
    to_sum_over = list(range(1, len(x.shape)))  # skip the first dimension 0 because that is the batch dimension

    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)

    with torch.no_grad():
        # (x!=fill) determines locations that might be unused, beause they are
        # missing the fill value we are looking for to indicate lack of use.
        # We then count the number of non-fill values over everything in that
        # time slot (reducing changes the shape to (B, T)). If any one entry
        # is non equal to this value, the item represent must be in use -
        # so return a value of true.
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask

class SmarterAttentionNet(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(SmarterAttentionNet, self).__init__()
        self.backbone = nn.Sequential(
            Flatten2(),  # Shape is now (B, T, D)
            nn.Linear(input_size, hidden_size),  # Shape becomes (B, T, neurons)
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )

        # Try changing this and see how the results change!
        self.score_net = AdditiveAttentionScore(hidden_size)
        self.apply_attn = ApplyAttention()

        self.prediction_net = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, input):
        mask = getMaskByFill(input)

        h = self.backbone(input)  # (B, T, D) -> (B, T, H)

        h_context = torch.mean(h, dim=1)  # (B, T, H) -> (B, H)

        scores = self.score_net(h, h_context)  # (B, T, H) , (B, H) -> (B, T, 1)

        final_context, _ = self.apply_attn(h, scores, mask=mask)

        return final_context