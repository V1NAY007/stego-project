"""
data.py
=======
Image loading + random-message generation for training, plus single-image I/O
helpers used at inference time.

Training uses random binary messages (the network must learn to carry *any*
bit string, not memorise one). Cover images come from a folder you point it at;
if the folder is empty/missing you can train on procedurally generated images
to smoke-test the pipeline without a dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: str, size: int | None = None) -> torch.Tensor:
    """Load an image file as a (3,H,W) float tensor in [0,1]."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a (3,H,W) float tensor in [0,1] as a LOSSLESS PNG.

    PNG matters: JPEG re-compression will destroy a payload unless you trained
    with a JPEG noise layer. Always distribute stego images as PNG.
    """
    arr = tensor.detach().clamp(0, 1).mul(255).round().byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, format="PNG")


def random_messages(batch: int, msg_len: int, device="cpu") -> torch.Tensor:
    """
    Random bits at a random per-image density.

    A fixed density of 0.5 leaves the model untrained for the messages it will
    actually be given: a framed payload is mostly zero padding (a 2-byte message
    in 256 bits is ~3% ones), and accuracy drops on exactly the inputs embed.py
    produces.
    """
    density = torch.rand(batch, 1, device=device) * 0.9 + 0.05
    return (torch.rand(batch, msg_len, device=device) < density).float()


class ImageFolder(Dataset):
    """
    size x size training samples, either by resizing the whole image or by
    taking a random crop at the image's NATIVE scale.

    crop=True matters more than it looks. Resizing a 640x480 photo down to
    128x128 destroys all pixel-scale detail, so a model trained that way has
    never seen the statistics of a full-resolution image -- and embedding into
    one at native resolution then falls apart (bit accuracy 0.999 -> 0.88).
    Cropping keeps the native pixel scale, so what the encoder learns to write
    is what it will be asked to write at inference.
    """

    def __init__(self, root: str, size: int = 128, crop: bool = False):
        self.size = size
        self.crop = crop
        self.paths = []
        if root and os.path.isdir(root):
            for p in Path(root).rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(p))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if not self.crop:
            return load_image(self.paths[idx], self.size)

        s = self.size
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.width < s or img.height < s:   # upscale only what is too small
            scale = s / min(img.width, img.height)
            img = img.resize((max(s, int(img.width * scale + 0.5)),
                              max(s, int(img.height * scale + 0.5))), Image.BICUBIC)
        x = np.random.randint(0, img.width - s + 1)
        y = np.random.randint(0, img.height - s + 1)
        arr = np.asarray(img.crop((x, y, x + s, y + s)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class SyntheticImages(Dataset):
    """
    Procedurally generated smooth+edgy images so you can verify the whole
    training loop end-to-end with zero external data.
    """

    def __init__(self, n: int = 512, size: int = 128, seed: int = 0):
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)
        s = self.size
        xs = np.linspace(0, 2 * np.pi, s)
        gx, gy = np.meshgrid(xs, xs)
        img = np.zeros((s, s, 3), dtype=np.float32)
        for c in range(3):
            fx, fy = rng.uniform(0.5, 4, 2)
            phase = rng.uniform(0, 2 * np.pi)
            base = 0.5 + 0.4 * np.sin(fx * gx + fy * gy + phase)
            base += 0.05 * rng.randn(s, s)  # a little texture
            img[..., c] = np.clip(base, 0, 1)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()
