import torch
import torch.nn as nn
from src.blocks.mbconv_block import MBConvBlock
from src.layers.conv_layer import ConvLayer
from src.layers.pooling_layers import GlobalAvgPool
from src.layers.flatten import Flatten
from src.layers.fc_layer import FCLayer

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.2):
        super().__init__()
        # Stem
        self.stem = ConvLayer(3, 32, kernel_size=3, stride=2)

        # MBConv stages (B0 baseline)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expansion_factor=1, kernel_size=3, stride=1),
            MBConvBlock(16, 24, expansion_factor=6, kernel_size=3, stride=2),
            MBConvBlock(24, 24, expansion_factor=6, kernel_size=3, stride=1),
            MBConvBlock(24, 40, expansion_factor=6, kernel_size=5, stride=2),
            MBConvBlock(40, 40, expansion_factor=6, kernel_size=5, stride=1),
            MBConvBlock(40, 80, expansion_factor=6, kernel_size=3, stride=2),
            MBConvBlock(80, 80, expansion_factor=6, kernel_size=3, stride=1),
            MBConvBlock(80, 112, expansion_factor=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expansion_factor=6, kernel_size=5, stride=1),
            MBConvBlock(112, 192, expansion_factor=6, kernel_size=5, stride=2),
            MBConvBlock(192, 192, expansion_factor=6, kernel_size=5, stride=1),
            MBConvBlock(192, 320, expansion_factor=6, kernel_size=3, stride=1)
        )

        # Head
        self.head_conv = ConvLayer(320, 1280, kernel_size=1)
        self.pool = GlobalAvgPool()
        self.flatten = Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = FCLayer(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
