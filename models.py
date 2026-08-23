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


# ---------------------------------------------------------------------------
# Native-resolution inference by tiling
# ---------------------------------------------------------------------------
# The nets are fully convolutional so they RUN at any resolution, but a model
# trained at S px only decodes reliably at S px: model_v2 goes 1.0000 -> 0.8625
# bit accuracy when the same image is embedded whole at 4K instead of 128px.
# The cause is the pixel scale of the residual, not the net's shape -- a
# well-optimised model has squeezed the residual to ~5/255 and has no margin
# left to survive being asked for a texture at a scale it never trained on.
#
# So don't ask it to. Cut the image into S x S tiles, embed the same message in
# each, reassemble: every tile is exactly in-distribution, the output is native
# resolution, and the repeated message across tiles is free redundancy on top of
# whatever --repeat does. Peak memory is one tile batch, not one 4K image.
#
# ponytail: tile borders leave a periodic residual step (measured 2.2/255 at a
# 128px period vs 0.96/255 inside a tile) -- invisible, but a fixed-period
# signature. If autocorrelation steganalysis ever matters, derive the grid
# offset from the key instead of pinning it to (0,0).


def _tiles(x: torch.Tensor, t: int):
    """(1,C,H,W) -> ((N,C,t,t) tiles, (nh,nw) grid), reflect-padded up to a multiple."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % t, (-w) % t          # both < t <= min(h,w), so reflect is legal
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    nh, nw = x.shape[-2] // t, x.shape[-1] // t
    return (x.view(1, -1, nh, t, nw, t).permute(0, 2, 4, 1, 3, 5)
             .reshape(nh * nw, -1, t, t)), (nh, nw)


def _untile(tiles: torch.Tensor, grid: tuple[int, int], h: int, w: int) -> torch.Tensor:
    """Inverse of _tiles, cropped back to the original (h,w)."""
    nh, nw = grid
    t = tiles.shape[-1]
    x = (tiles.view(1, nh, nw, -1, t, t).permute(0, 3, 1, 4, 2, 5)
              .reshape(1, -1, nh * t, nw * t))
    return x[..., :h, :w]


def _batched(fn, tiles: torch.Tensor, batch: int) -> torch.Tensor:
    return torch.cat([fn(tiles[i:i + batch]) for i in range(0, len(tiles), batch)])


@torch.no_grad()
def tiled_encode(encoder: nn.Module, cover: torch.Tensor, msg: torch.Tensor,
                 tile: int, batch: int = 16) -> torch.Tensor:
    """(1,3,H,W) cover + (1,L) msg -> (1,3,H,W) stego, embedded tile-by-tile."""
    h, w = cover.shape[-2:]
    if min(h, w) < tile:                 # too small to tile: one whole-image pass
        return encoder(cover, msg)
    tiles, grid = _tiles(cover, tile)
    out = _batched(lambda x: encoder(x, msg.expand(x.shape[0], -1)), tiles, batch)
    return _untile(out, grid, h, w)


@torch.no_grad()
def tiled_decode(decoder: nn.Module, stego: torch.Tensor, tile: int,
                 batch: int = 16) -> torch.Tensor:
    """(1,3,H,W) stego -> (L,) bit PROBABILITIES, soft-averaged over all tiles."""
    h, w = stego.shape[-2:]
    if min(h, w) < tile:
        return torch.sigmoid(decoder(stego))[0]
    tiles, (nh, nw) = _tiles(stego, tile)
    # Vote only with tiles that lie WHOLLY inside the image. The reflect-padded
    # edge tiles had part of their residual cropped off when the stego was saved,
    # so they read noise -- averaging them in just dilutes the good tiles.
    keep = [r * nw + c for r in range(h // tile) for c in range(w // tile)]
    probs = _batched(lambda x: torch.sigmoid(decoder(x)), tiles[keep], batch)
    return probs.mean(0)


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
        # 3. tiling must round-trip geometry exactly, at sizes that do NOT
        #    divide the tile, and must not silently change the output shape.
        for h, w in [(128, 128), (300, 200), (129, 257)]:
            big = torch.rand(1, 3, h, w)
            t, g = _tiles(big, 128)
            assert torch.equal(_untile(t, g, h, w), big), (h, w)
            s = tiled_encode(enc, big, msg[:1], 128)
            assert s.shape == big.shape, (s.shape, big.shape)
            assert tiled_decode(dec, s, 128).shape == (msg_len,)
    print(f"ok: bit locality max/mean {d.max()/d.mean():.1f}, "
          f"{feat.shape[1]} decoder features -> {msg_len} bits "
          f"(grid {enc.grid}x{enc.grid}), tiling round-trips")
