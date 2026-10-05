"""Shared convolutional feature extractor.

Both the softmax baseline and the Bayesian last layer use this network, so
the comparison is about the classifier head rather than the architecture.
The final batch-norm keeps features on a stable scale for the last-layer prior.
"""

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        return self.net(x)
