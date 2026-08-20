"""
models.py
=========
The convolutional networks. This is the part that makes the scheme *not* LSB.

Encoder:  cover image + message bits  ->  stego image
    The message is tiled across the spatial grid and fused with image features.
    The network outputs a *bounded residual* that is ADDED to the cover image.
    Because the residual is produced by conv filters trained jointly with an
    image-distortion loss, the secret ends up as a faint, natural-image-shaped
    texture spread over the WHOLE image and across ALL bit planes -- not stuffed
    into the least-significant bits. That is precisely why LSB/RS/chi-square
    steganalysis (which model LSB-flip statistics) do not flag it.

Decoder:  stego image  ->  recovered message bits
    A second CNN that has learned the inverse mapping. Without these exact
    weights (and the key), the payload is unrecoverable.

Discriminator (optional, GAN-style):  image -> real/stego logit
    Trains the encoder to make stego images statistically indistinguishable
    from covers, pushing distortion into imperceptible directions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """cover (B,3,H,W) in [0,1] + msg (B,L) in {0,1} -> stego (B,3,H,W) in [0,1]."""

    def __init__(self, msg_len: int, hidden: int = 64, n_blocks: int = 4,
                 residual_scale: float = 0.1):
        super().__init__()
        self.msg_len = msg_len
        self.residual_scale = residual_scale

        self.pre = conv_bn_relu(3, hidden)

        # after concatenating [image features | tiled msg | raw cover]
        in_ch = hidden + msg_len + 3
        blocks = [conv_bn_relu(in_ch, hidden)]
        for _ in range(n_blocks - 1):
            # re-inject the message at every block (HiDDeN trick) for stronger signal
            blocks.append(conv_bn_relu(hidden + msg_len, hidden))
        self.blocks = nn.ModuleList(blocks)

        self.to_residual = nn.Conv2d(hidden, 3, kernel_size=1)

    def forward(self, cover: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        b, _, h, w = cover.shape
        feat = self.pre(cover)
        msg_map = msg.view(b, self.msg_len, 1, 1).expand(b, self.msg_len, h, w)

        x = torch.cat([feat, msg_map, cover], dim=1)
        x = self.blocks[0](x)
        for block in self.blocks[1:]:
            x = block(torch.cat([x, msg_map], dim=1))

        residual = torch.tanh(self.to_residual(x)) * self.residual_scale
        stego = torch.clamp(cover + residual, 0.0, 1.0)
        return stego


class Decoder(nn.Module):
    """stego (B,3,H,W) -> message logits (B,L). Apply sigmoid for probabilities."""

    def __init__(self, msg_len: int, hidden: int = 64, n_blocks: int = 6):
        super().__init__()
        layers = [conv_bn_relu(3, hidden)]
        for _ in range(n_blocks - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.to_msg = nn.Conv2d(hidden, msg_len, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x = self.body(stego)
        x = self.to_msg(x)
        x = self.pool(x).flatten(1)  # (B, L) logits
        return x


class Discriminator(nn.Module):
    """image -> single logit; high = 'looks like a stego / has a payload'."""

    def __init__(self, hidden: int = 64, n_blocks: int = 3):
        super().__init__()
        layers = [conv_bn_relu(3, hidden)]
        for _ in range(n_blocks - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.body(img)
        x = self.pool(x).flatten(1)
        return self.head(x)  # logit


def build_models(msg_len: int, hidden: int = 64, residual_scale: float = 0.1):
    """Convenience factory."""
    return (
        Encoder(msg_len, hidden=hidden, residual_scale=residual_scale),
        Decoder(msg_len, hidden=hidden),
        Discriminator(hidden=hidden),
    )
