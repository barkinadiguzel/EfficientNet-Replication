import torch
import torch.nn as nn
from src.layers.conv_layer import ConvLayer
from src.layers.depthwise_conv import DepthwiseConv
from src.layers.se_layer import SELayer

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=1, reduction=4):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        hidden_dim = in_channels * expansion_factor

        # 1. Expansion phase (pointwise conv)
        self.expand_conv = ConvLayer(in_channels, hidden_dim, kernel_size=1)

        # 2. Depthwise conv
        self.dw_conv = DepthwiseConv(hidden_dim, kernel_size=kernel_size, stride=stride)

        # 3. Squeeze-and-Excitation
        self.se = SELayer(hidden_dim, reduction=reduction)

        # 4. Projection phase (pointwise conv)
        self.project_conv = ConvLayer(hidden_dim, out_channels, kernel_size=1, activation=False)

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.dw_conv(out)
        out = self.se(out)
        out = self.project_conv(out)

        if self.use_residual:
            out = out + x  

        return out
