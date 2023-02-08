import torch
from torch import nn


class ImplicitFunction(nn.Module):

    def __init__(self, config, params=None):
        super().__init__()
        self.params = params
        self.relu = nn.LeakyReLU(0.2)
        # self.dropout = nn.Dropout(0.5)
        self.hidden_dim = config.hidden_dim

    def set_params(self, params):
        self.params = params

    def forward(self, points, params=None):
        if params is not None:
            self.params = params

        if self.params is None:
            raise ValueError('Can not run forward on uninitialized implicit function')

        x = points
        # TODO: I just added unsqueeze(1), reshape(-1) and bmm and everything works (or did I introduce some kind of bug?)
        weights, scales, biases = self.params[0]
        weights = weights.reshape(-1, 3, self.hidden_dim)
        scales = scales.unsqueeze(1)
        biases = biases.unsqueeze(1)

        x = torch.bmm(x, weights) * scales + biases
        # x = self.dropout(x)
        x = self.relu(x)

        for layer in self.params[1:-1]:
            weights, scales, biases = layer

            weights = weights.reshape(-1, self.hidden_dim, self.hidden_dim)
            scales = scales.unsqueeze(1)
            biases = biases.unsqueeze(1)

            x = torch.bmm(x, weights) * scales + biases
            # x = self.dropout(x)
            x = self.relu(x)

        weights, scales, biases = self.params[-1]

        weights = weights.reshape(-1, self.hidden_dim, 1)
        scales = scales.unsqueeze(1)
        biases = biases.unsqueeze(1)

        x = torch.bmm(x, weights) * scales + biases

        return x