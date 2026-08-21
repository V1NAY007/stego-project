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
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_models
from noise_layers import NoisePool
from data import ImageFolder, SyntheticImages, random_messages


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
    p.add_argument("--img-weight", type=float, default=0.7)
    p.add_argument("--img-warmup", type=int, default=5,
                   help="epochs to ramp the image loss up from 0. At init the "
                        "decoder is random, so the message loss is flat while the "
                        "image loss can be driven to ~0 by shrinking the residual "
                        "below one 8-bit step -- after which quantization erases "
                        "it. Ramping lets the decoder learn first. 0 = off.")
    p.add_argument("--adv-weight", type=float, default=0.01)
    p.add_argument("--no-noise", action="store_true", help="disable robustness noise")
    p.add_argument("--out", type=str, default="checkpoints/model.pt")
    p.add_argument("--workers", type=int, default=2,
                   help="dataloader workers; set near the vCPU count. Measured on "
                        "COCO at 128px: 4 workers decode ~834 img/s while the step "
                        "itself runs at ~70-200 img/s, so this is not the "
                        "bottleneck -- 4 is plenty and more buys nothing.")
    p.add_argument("--amp", action="store_true",
                   help="fp16 autocast + channels_last. ~2x on Ampere and later. "
                        "OFF by default on purpose: the payload lives at ~10/255, "
                        "so reduced precision is not free here. Losses are kept in "
                        "fp32 regardless, but verify bit-acc and PSNR match a "
                        "non-amp run before trusting a long one.")
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
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.workers > 0)

    encoder, decoder, disc = build_models(
        args.msg_len, hidden=args.hidden, residual_scale=args.residual_scale)
    encoder, decoder, disc = encoder.to(device), decoder.to(device), disc.to(device)
    noise = NoisePool(enabled=not args.no_noise).to(device)

    cuda = device.type == "cuda"
    amp_on = args.amp and cuda
    # input shapes are fixed for the whole run, so let cuDNN pick its best algos
    torch.backends.cudnn.benchmark = cuda
    if amp_on:
        encoder = encoder.to(memory_format=torch.channels_last)
        decoder = decoder.to(memory_format=torch.channels_last)
        disc = disc.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    def autocast():
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on)

    print(f"amp={amp_on} channels_last={amp_on} cudnn.benchmark={cuda} "
          f"workers={args.workers} batch={args.batch}")

    opt_ed = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save():
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

    for epoch in range(args.epochs):
        encoder.train(); decoder.train(); disc.train()
        # ponytail: linear ramp; cosine only if this ever needs tuning
        img_w = args.img_weight * min(1.0, (epoch + 1) / max(args.img_warmup, 1))
        # accumulate on the GPU. Calling .item() per step forces a sync and stalls
        # the pipeline -- it cost ~40% of throughput (43 img/s in the loop vs
        # 70 img/s for the same step in isolation). One sync per epoch instead.
        agg = {k: torch.zeros((), device=device) for k in ("msg", "mse", "acc")}
        n = 0

        for cover in loader:
            cover = cover.to(device, non_blocking=True)
            if amp_on:
                cover = cover.contiguous(memory_format=torch.channels_last)
            b = cover.size(0)
            msg = random_messages(b, args.msg_len, device)

            # ONE encoder forward per step. The old code ran a second one for the
            # generator update; the discriminator only ever sees stego detached, so
            # it is not part of loss_d's graph and loss_d.backward() cannot free it.
            # The extra forward was pure waste (~20% of the step).
            with autocast():
                stego = encoder(cover, msg)

            # ---- train discriminator ----
            with autocast():
                d_real = disc(cover).float()
                d_fake = disc(stego.detach()).float()
            loss_d = (bce(d_real, torch.zeros_like(d_real))
                      + bce(d_fake, torch.ones_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)

            # ---- train encoder + decoder ----
            with autocast():
                logits = decoder(noise(stego, cover)).float()
                d_on_stego = disc(stego).float()

            # Losses in fp32. The payload lives at ~10/255 and the whole point of
            # `res` is sub-1/255 behaviour, so this is not a place to give up
            # mantissa bits -- only the convolutions run reduced precision.
            loss_msg = bce(logits, msg)
            loss_img = mse(stego.float(), cover)
            loss_adv = bce(d_on_stego, torch.zeros_like(d_on_stego))  # fool disc

            loss = (args.msg_weight * loss_msg
                    + img_w * loss_img
                    + args.adv_weight * loss_adv)
            opt_ed.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt_ed)
            scaler.update()

            with torch.no_grad():
                agg["acc"] += ((logits > 0.0).float() == msg).float().mean() * b
                agg["msg"] += loss_msg.detach() * b
                agg["mse"] += loss_img.detach() * b
                n += b

        n = max(n, 1)
        acc = (agg["acc"] / n).item()
        msg_loss = (agg["msg"] / n).item()
        mean_mse = (agg["mse"] / n).item()
        # PSNR of the mean MSE, which is the standard definition (the old code
        # averaged per-batch PSNRs, which is not the same number).
        psnr_db = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))
        # residual RMS in 8-bit steps. Below ~1.0 the payload does not survive
        # PNG rounding -- watch this, not just PSNR.
        res = math.sqrt(mean_mse) * 255.0
        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"bit-acc {acc:.4f} | PSNR {psnr_db:5.2f} dB | "
              f"res {res:4.2f}/255 | img_w {img_w:.2f} | "
              f"msg {msg_loss:.4f} | img {mean_mse:.5f}")
        # every epoch, not just the last: an epoch here can be 40+ minutes, and a
        # reclaimed spot instance or an OOM would otherwise cost the whole run.
        save()

    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
