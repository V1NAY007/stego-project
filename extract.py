"""
extract.py
==========
Recover a hidden message from a stego image using the trained decoder + key.
The correct key is required -- a wrong key yields scrambled bits -> garbage.

    python extract.py --model checkpoints/model.pt \
        --stego stego.png --key "hunter2" --repeat 4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from models import Decoder
from data import load_image
from payload import decode_payload
from crypto_key import unpermute_bits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stego", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--raw-out", default=None,
                   help="if set, write recovered bytes here instead of printing text")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["config"]
    decoder = Decoder(cfg["msg_len"], hidden=cfg["hidden"]).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    stego = load_image(args.stego, size=cfg["size"]).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(decoder(stego))[0].cpu().numpy()

    bits = unpermute_bits(probs, args.key)  # undo key layer (on soft values)
    data = decode_payload(bits, repeat=args.repeat)

    if args.raw_out:
        with open(args.raw_out, "wb") as f:
            f.write(data)
        print(f"wrote {len(data)} bytes -> {args.raw_out}")
    else:
        try:
            print("recovered message:", data.decode("utf-8"))
        except UnicodeDecodeError:
            print("recovered raw bytes (not valid UTF-8):", data)


if __name__ == "__main__":
    main()
