import torch
import torch.nn as nn

class AdaptiveAvgPoolLayer(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)
