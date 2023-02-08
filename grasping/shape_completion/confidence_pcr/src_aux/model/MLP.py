import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = (input_size + output_size) // 2
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)