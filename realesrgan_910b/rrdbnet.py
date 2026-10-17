"""Inference-only x4 RRDBNet, compatible with RealESRGAN_x4plus.pth.

Adapted from BasicSR v1.4.2 basicsr/archs/rrdbnet_arch.py.
Copyright 2018-2022 BasicSR Authors. Apache-2.0 (see LICENSE-BasicSR.txt).
Changes: x4 only; no training initialization, registry, or BasicSR dependency.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, features=64, growth=32):
        super().__init__()
        for i in range(1, 6):
            setattr(self, f"conv{i}", nn.Conv2d(
                features + (i - 1) * growth, features if i == 5 else growth, 3, 1, 1))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        features = [x]
        for i in range(1, 5):
            features.append(self.lrelu(getattr(self, f"conv{i}")(torch.cat(features, dim=1))))
        return x + self.conv5(torch.cat(features, dim=1)) * 0.2


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, x):
        return x + self.rdb3(self.rdb2(self.rdb1(x))) * 0.2


class RRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv_first(x)
        x = x + self.conv_body(self.body(x))
        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(x)))
