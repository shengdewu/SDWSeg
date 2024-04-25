import torch.nn as nn
import torch

__all__ = [
    'SELayer'
]


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Module
    """
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, in_channels, 1, bias=True)
        return

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x * torch.sigmoid(y)
