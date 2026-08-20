"""
train.py
========
Jointly train the encoder + decoder (+ discriminator) so that:
  * the decoder recovers the message   (message loss, drives capacity)
  * the stego looks like the cover      (image loss, drives imperceptibility)
  * a discriminator can't tell them apart (adversarial loss, hides statistics)

Usage
-----
Real images:
    python train.py --data /path/to/images --size 128 --msg-len 256 --epochs 40

Smoke test with no dataset (synthetic images, fast):
    python train.py --synthetic --size 64 --msg-len 64 --epochs 2 --batch 8

Output: checkpoints/model.pt  (contains encoder+decoder weights and config)
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_models
from noise_layers import NoisePool
from data import ImageFolder, SyntheticImages, random_messages


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="")
    p.add_argument("--synthetic", action="store_true",
                   help="train on procedurally generated images (no dataset needed)")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--msg-len", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--msg-weight", type=float, default=1.0)
    p.add_argument("--img-weight", type=float, default=2.0)
    p.add_argument("--adv-weight", type=float, default=0.01)
    p.add_argument("--no-noise", action="store_true", help="disable robustness noise")
    p.add_argument("--out", type=str, default="checkpoints/model.pt")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"device={device}")

    if args.synthetic or not args.data:
        ds = SyntheticImages(n=512, size=args.size)
        print("using SYNTHETIC images")
    else:
        ds = ImageFolder(args.data, size=args.size)
        if len(ds) == 0:
            raise SystemExit(f"no images found in {args.data}")
        print(f"found {len(ds)} images")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=2, drop_last=True)

    encoder, decoder, disc = build_models(
        args.msg_len, hidden=args.hidden, residual_scale=args.residual_scale)
    encoder, decoder, disc = encoder.to(device), decoder.to(device), disc.to(device)
    noise = NoisePool(enabled=not args.no_noise).to(device)

    opt_ed = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        encoder.train(); decoder.train(); disc.train()
        agg = {"msg": 0.0, "img": 0.0, "acc": 0.0, "psnr": 0.0, "n": 0}

        for cover in loader:
            cover = cover.to(device)
            b = cover.size(0)
            msg = random_messages(b, args.msg_len, device)

            # ---- train discriminator ----
            stego = encoder(cover, msg)
            d_real = disc(cover)
            d_fake = disc(stego.detach())
            loss_d = bce(d_real, torch.zeros_like(d_real)) + \
                     bce(d_fake, torch.ones_like(d_fake))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- train encoder + decoder ----
            stego = encoder(cover, msg)
            noised = noise(stego, cover)
            logits = decoder(noised)

            loss_msg = bce(logits, msg)
            loss_img = mse(stego, cover)
            d_on_stego = disc(stego)
            loss_adv = bce(d_on_stego, torch.zeros_like(d_on_stego))  # fool disc

            loss = (args.msg_weight * loss_msg
                    + args.img_weight * loss_img
                    + args.adv_weight * loss_adv)
            opt_ed.zero_grad(); loss.backward(); opt_ed.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == msg).float().mean().item()
                agg["msg"] += loss_msg.item() * b
                agg["img"] += loss_img.item() * b
                agg["acc"] += acc * b
                agg["psnr"] += psnr(stego, cover) * b
                agg["n"] += b

        n = max(agg["n"], 1)
        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"bit-acc {agg['acc']/n:.4f} | PSNR {agg['psnr']/n:5.2f} dB | "
              f"msg {agg['msg']/n:.4f} | img {agg['img']/n:.5f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "config": {
            "msg_len": args.msg_len,
            "hidden": args.hidden,
            "residual_scale": args.residual_scale,
            "size": args.size,
        },
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
