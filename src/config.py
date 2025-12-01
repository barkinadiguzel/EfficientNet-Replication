# config.py
# EfficientNet-B0 Baseline Configuration

# Input image size
INPUT_SIZE = 224

# Number of output classes
NUM_CLASSES = 1000  # ImageNet

# Dropout rate for final FC layer
DROPOUT_RATE = 0.2

# MBConv block expansion factors and repeats (B0 baseline)
# (operator_type, kernel_size, repeats, input_channels, output_channels, stride, expansion_factor)
MB_CONV_SETTINGS = [
    ("MBConv1", 3, 1, 32, 16, 1, 1),    # Stage 1
    ("MBConv6", 3, 2, 16, 24, 2, 6),    # Stage 2
    ("MBConv6", 5, 2, 24, 40, 2, 6),    # Stage 3
    ("MBConv6", 3, 3, 40, 80, 2, 6),    # Stage 4
    ("MBConv6", 5, 3, 80, 112, 1, 6),   # Stage 5
    ("MBConv6", 5, 4, 112, 192, 2, 6),  # Stage 6
    ("MBConv6", 3, 1, 192, 320, 1, 6),  # Stage 7
]

# Scaling coefficients for compound scaling (example, B0 baseline)
ALPHA = 1.0  # depth
BETA  = 1.0  # width
GAMMA = 1.0  # resolution
PHI   = 0    # scaling factor, 0 for B0
