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

from models import Decoder, tiled_decode
from data import load_image
from payload import decode_payload
from crypto_key import unpermute_bits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stego", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--size", type=int, default=0,
                   help="0 (default) reads the stego at its NATIVE resolution, "
                        "which is what you want -- the file already carries "
                        "whatever size it was embedded at, so this needs no "
                        "coordination with embed.py. Only set it if you resized "
                        "the file yourself (which will likely lose the payload).")
    p.add_argument("--tile", type=int, default=0,
                   help="must match embed.py. 0 (default) = the model's training "
                        "size from the checkpoint, same default embed.py uses.")
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

    stego = load_image(args.stego, size=args.size or None).unsqueeze(0).to(device)
    # Every tile carries the same message, so averaging their probabilities is
    # free error correction on top of --repeat: 480 votes on a 4K image.
    probs = tiled_decode(decoder, stego,
                         args.tile or cfg.get("size", 128)).cpu().numpy()

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
