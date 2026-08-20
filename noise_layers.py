"""
noise_layers.py
===============
Differentiable image corruptions inserted BETWEEN encoder and decoder during
training. Training through them teaches the decoder to survive real-world
handling (re-saving, mild compression, resizing noise). This also further
decorrelates the payload from any single bit plane -> even harder for LSB tools.

All layers keep values in [0,1] and are (sub-)differentiable so gradients flow
back to the encoder.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.02):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.sigma <= 0:
            return x
        return torch.clamp(x + torch.randn_like(x) * self.sigma, 0.0, 1.0)


class Quantize(nn.Module):
    """Simulate 8-bit rounding with a straight-through estimator."""

    def forward(self, x):
        q = torch.round(x * 255.0) / 255.0
        return x + (q - x).detach()  # forward = quantized, backward = identity


class Dropout(nn.Module):
    """Randomly replace some stego pixels with the original cover pixels."""

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, stego, cover):
        mask = (torch.rand_like(stego[:, :1]) < self.p).float()
        return stego * (1 - mask) + cover * mask


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        ax = torch.arange(kernel_size) - kernel_size // 2
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k = torch.outer(g, g)
        k = (k / k.sum()).view(1, 1, kernel_size, kernel_size)
        self.register_buffer("kernel", k)

    def forward(self, x):
        k = self.kernel.expand(x.shape[1], 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, k, padding=self.kernel_size // 2, groups=x.shape[1])


class NoisePool(nn.Module):
    """
    Picks one random corruption per batch. Dropout needs the cover, so we pass
    it through. Keep JPEG out of this minimal build; add a differentiable-JPEG
    layer (e.g. DiffJPEG) here later if you need JPEG robustness.
    """

    def __init__(self, sigma: float = 0.02, dropout_p: float = 0.3,
                 enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.identity = Identity()
        self.gauss = GaussianNoise(sigma)
        self.quant = Quantize()
        self.blur = GaussianBlur()
        self.dropout = Dropout(dropout_p)

    def forward(self, stego, cover):
        if not self.enabled:
            return self.quant(stego)
        choice = random.choice(["identity", "gauss", "quant", "blur", "dropout"])
        if choice == "identity":
            return self.identity(stego)
        if choice == "gauss":
            return self.gauss(stego)
        if choice == "quant":
            return self.quant(stego)
        if choice == "blur":
            return torch.clamp(self.blur(stego), 0.0, 1.0)
        return self.dropout(stego, cover)
