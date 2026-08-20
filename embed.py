"""
embed.py
========
Hide a secret message inside a cover image using a trained model + a key.

    python embed.py --model checkpoints/model.pt \
        --cover cover.png --out stego.png \
        --message "meet at the docks, midnight" --key "hunter2" --repeat 4

The output MUST be saved as PNG (done automatically). The message is:
  1. framed with a length header,
  2. repetition-coded (--repeat) for robustness,
  3. permuted by the key,
  4. embedded by the encoder CNN as a faint whole-image residual.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from models import Encoder
from data import load_image, save_image
from payload import encode_payload
from crypto_key import permute_bits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cover", required=True)
    p.add_argument("--out", default="stego.png")
    p.add_argument("--message", default=None, help="UTF-8 text to hide")
    p.add_argument("--message-file", default=None, help="read raw bytes from file")
    p.add_argument("--key", required=True)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["config"]
    encoder = Encoder(cfg["msg_len"], hidden=cfg["hidden"],
                      residual_scale=cfg["residual_scale"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    if args.message_file:
        with open(args.message_file, "rb") as f:
            data = f.read()
    elif args.message is not None:
        data = args.message.encode("utf-8")
    else:
        raise SystemExit("provide --message or --message-file")

    bits = encode_payload(data, cfg["msg_len"], repeat=args.repeat)
    bits = permute_bits(bits, args.key)  # key layer
    msg = torch.from_numpy(bits.astype(np.float32)).unsqueeze(0).to(device)

    cover = load_image(args.cover, size=cfg["size"]).unsqueeze(0).to(device)

    with torch.no_grad():
        stego = encoder(cover, msg)

    save_image(stego[0], args.out)

    mse = torch.mean((stego - cover) ** 2).item()
    psnr = 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)
    print(f"embedded {len(data)} bytes -> {args.out}")
    print(f"PSNR vs cover: {psnr:.2f} dB (higher = more invisible)")
    print("Distribute as PNG. Re-saving as JPEG will likely destroy the payload.")


if __name__ == "__main__":
    main()
