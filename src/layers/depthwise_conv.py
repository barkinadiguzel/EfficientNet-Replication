import torch
import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.bn(self.depthwise(x)))
