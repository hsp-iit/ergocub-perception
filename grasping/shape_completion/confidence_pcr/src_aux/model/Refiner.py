import copy

import torch
from torch import nn, autograd
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD

from model.ImplicitFunction import ImplicitFunction


class Refiner(nn.Module):

    def __init__(self):
        super().__init__()

        class Config:
            hidden_dim = 32

        self.sdf = ImplicitFunction(Config)

    def forward(self, points, weights):

        loss_function = BCEWithLogitsLoss(reduction='mean')
        points = points.clone().detach().requires_grad_(True)

        results = self.sdf(points, weights)

        gt = torch.ones_like(results[..., 0], dtype=torch.float32)
        loss = loss_function(results[..., 0], gt)
        grad = autograd.grad(loss, points)
        points = points - (0.1 * grad[0])

        results = self.sdf(points, weights)
        idx = torch.sigmoid(results) >= 0.5
        return points, idx
