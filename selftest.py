"""
selftest.py
===========
Fast end-to-end check that the whole pipeline is wired correctly, using
synthetic images and a tiny model. Runs on CPU in ~1-2 minutes.

    python selftest.py

It trains briefly, embeds a real text message with a key, extracts it back,
and confirms a wrong key fails. Bit accuracy should climb well above 0.5;
with enough steps the exact message round-trips. This is a wiring test, not a
quality benchmark -- real quality needs real images and more epochs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models import build_models
from noise_layers import NoisePool
from data import SyntheticImages, random_messages
from payload import encode_payload, decode_payload
from crypto_key import permute_bits, unpermute_bits


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    size, msg_len, hidden = 48, 64, 32
    encoder, decoder, disc = build_models(msg_len, hidden=hidden, residual_scale=0.15)
    noise = NoisePool(enabled=True)

    ds = SyntheticImages(n=64, size=size)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=2e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    steps = 0
    for epoch in range(15):
        for cover in loader:
            msg = random_messages(cover.size(0), msg_len)
            stego = encoder(cover, msg)
            logits = decoder(noise(stego, cover))
            loss = bce(logits, msg) + 2.0 * mse(stego, cover)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) > 0.5).float() == msg).float().mean().item()
        print(f"epoch {epoch+1:2d} | steps {steps:3d} | bit-acc {acc:.3f}")

    # ---- real message round trip ----
    encoder.eval(); decoder.eval()
    key = "correct horse battery staple"
    text = "hi"  # 2 bytes -> 16 payload bits + 16 header = 32 bits; fits with repeat=1
    bits = encode_payload(text.encode(), msg_len, repeat=1)
    bits_key = permute_bits(bits, key)
    msg = torch.from_numpy(bits_key.astype(np.float32)).unsqueeze(0)

    cover = ds[0].unsqueeze(0)
    with torch.no_grad():
        stego = encoder(cover, msg)
        probs = torch.sigmoid(decoder(stego))[0].numpy()

    recovered = decode_payload(unpermute_bits(probs, key), repeat=1)
    wrong = decode_payload(unpermute_bits(probs, key + "X"), repeat=1)

    raw_bit_acc = float(((probs > 0.5).astype(np.uint8) == bits_key.astype(np.uint8)).mean())
    print("\n--- round trip ---")
    print(f"raw bit accuracy       : {raw_bit_acc:.3f}")
    print(f"recovered (right key)  : {recovered!r}")
    print(f"recovered (wrong key)  : {wrong[:16]!r}")
    print("\nselftest finished. (Short training => may not be bit-perfect; "
          "accuracy should be well above 0.5 and rising.)")


if __name__ == "__main__":
    main()
