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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def msg_grid(msg_len: int) -> int:
    """Side of the smallest square grid that holds msg_len bits, one bit per cell."""
    return int(math.ceil(math.sqrt(msg_len)))


class Encoder(nn.Module):
    """cover (B,3,H,W) in [0,1] + msg (B,L) in {0,1} -> stego (B,3,H,W) in [0,1]."""

    def __init__(self, msg_len: int, hidden: int = 64, n_blocks: int = 4,
                 residual_scale: float = 0.1):
        super().__init__()
        self.msg_len = msg_len
        self.grid = msg_grid(msg_len)
        self.residual_scale = residual_scale

        self.pre = conv_bn_relu(3, hidden)

        # after concatenating [image features | msg map (1ch) | raw cover]
        in_ch = hidden + 1 + 3
        blocks = [conv_bn_relu(in_ch, hidden)]
        for _ in range(n_blocks - 1):
            # re-inject the message at every block (HiDDeN trick) for stronger signal
            blocks.append(conv_bn_relu(hidden + 1, hidden))
        self.blocks = nn.ModuleList(blocks)

        self.to_residual = nn.Conv2d(hidden, 3, kernel_size=1)

    def _spread(self, msg: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        (B,L) bits -> (B,1,H,W) map where bit i owns one cell of a grid x grid tiling.

        The old version tiled ALL L bits at EVERY position, as L channels. But a
        stack of translation-equivariant convolutions fed a spatially CONSTANT
        input can only produce a spatially constant output -- so the encoder's
        clean message channel was a 3-number global RGB offset, and every bit
        beyond that had to be smuggled through cover-dependent modulation for the
        decoder to disentangle. That is the actual capacity ceiling: 8 bits
        trained, 64 crawled, 256 sat at chance, and residual amplitude was never
        the constraint. One cell per bit removes it.
        """
        b, g = msg.shape[0], self.grid
        x = msg * 2.0 - 1.0                       # {0,1} -> {-1,+1}: a 0 bit is
        pad = g * g - self.msg_len                # signal, not absence of signal
        if pad:
            x = torch.cat([x, x.new_zeros(b, pad)], dim=1)
        x = x.view(b, 1, g, g)
        return F.interpolate(x, size=(h, w), mode="nearest")

    def forward(self, cover: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        _, _, h, w = cover.shape
        feat = self.pre(cover)
        msg_map = self._spread(msg, h, w)

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
        self.msg_len = msg_len
        self.grid = msg_grid(msg_len)
        layers = [conv_bn_relu(3, hidden)]
        for _ in range(n_blocks - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        # Pool to the SAME grid the encoder wrote on, so each cell reads its own
        # bit from its own `hidden` features. The old head was
        # Conv2d(hidden, msg_len, 1) -> AdaptiveAvgPool2d(1); a 1x1 conv is
        # per-position linear and average pooling is linear, so the two commute
        # and the head only ever saw `hidden` (=64) numbers for the whole
        # message. The 3x3 layer below lets a cell use its neighbours to cancel
        # the crosstalk the body's receptive field introduces.
        self.pool = nn.AdaptiveAvgPool2d(self.grid)
        self.head = nn.Sequential(conv_bn_relu(hidden, hidden),
                                  nn.Conv2d(hidden, 1, kernel_size=1))

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.body(stego))
        x = self.head(x).flatten(1)      # (B, grid*grid)
        return x[:, :self.msg_len]       # (B, L) logits


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


if __name__ == "__main__":
    # Regression guards for the two capacity ceilings. Run: python models.py
    torch.manual_seed(0)
    msg_len, hidden, b = 256, 64, 4
    enc, dec, _ = build_models(msg_len, hidden=hidden)
    enc.eval(); dec.eval()
    cover = torch.rand(b, 3, 128, 128)
    msg = (torch.rand(b, msg_len) > 0.5).float()
    with torch.no_grad():
        stego = enc(cover, msg)
        assert stego.shape == cover.shape, stego.shape
        assert dec(stego).shape == (b, msg_len)

        # 1. one bit must move a LOCAL region. A spatially constant msg_map through
        #    translation-equivariant convs can only move the whole image at once.
        flipped = msg.clone()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        d = (enc(cover, flipped) - stego).abs().mean(dim=(0, 1))
        assert d.max() > 5 * d.mean(), (
            f"message is not spatially localised (max/mean {d.max()/d.mean():.1f})")

        # 2. the message head must see more than `hidden` numbers for the whole msg.
        feat = dec.pool(dec.body(stego)).flatten(1)
        assert feat.shape[1] >= msg_len, (
            f"decoder bottleneck is back: {feat.shape[1]} features for {msg_len} bits")
    print(f"ok: bit locality max/mean {d.max()/d.mean():.1f}, "
          f"{feat.shape[1]} decoder features -> {msg_len} bits "
          f"(grid {enc.grid}x{enc.grid})")
