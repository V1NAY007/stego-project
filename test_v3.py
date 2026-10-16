"""
test_v3.py
==========
Regression tests for the 496-bit (60-byte) capacity upgrade.
Validates architecture, payload, training loop, and checkpoint round-trip.

    python test_v3.py

All tests use CPU and no dataset. This is an architectural compatibility check,
not a quality benchmark -- actual message recovery requires trained weights.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from models import Encoder, Decoder, Discriminator, build_models, msg_grid
from models import tiled_encode, tiled_decode, _tiles, _untile
from noise_layers import NoisePool
from data import SyntheticImages, random_messages
from payload import encode_payload, decode_payload, HEADER_BITS
from crypto_key import permute_bits, unpermute_bits


def test_msg_grid():
    """Grid sizes for key msg_len values."""
    assert msg_grid(256) == 16
    assert msg_grid(416) == 21
    assert msg_grid(496) == 23
    assert msg_grid(64) == 8
    print("  msg_grid: ok")


def test_encoder_shapes():
    """Encoder accepts 496-bit messages and produces correct output shape."""
    for msg_len in (256, 416, 496):
        enc = Encoder(msg_len, hidden=64)
        enc.eval()
        cover = torch.rand(2, 3, 128, 128)
        msg = torch.rand(2, msg_len)
        with torch.no_grad():
            stego = enc(cover, msg)
        assert stego.shape == cover.shape, f"msg_len={msg_len}: {stego.shape}"
    print("  encoder shapes: ok")


def test_decoder_shapes():
    """Decoder produces exactly msg_len logits."""
    for msg_len in (256, 416, 496):
        dec = Decoder(msg_len, hidden=64)
        dec.eval()
        stego = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            logits = dec(stego)
        assert logits.shape == (2, msg_len), f"msg_len={msg_len}: {logits.shape}"
        grid = msg_grid(msg_len)
        feat = dec.pool(dec.body(stego)).flatten(1)
        assert feat.shape[1] == grid * grid * 64
    print("  decoder shapes: ok")


def test_discriminator_unchanged():
    """Discriminator is msg_len-independent."""
    disc = Discriminator(hidden=64)
    disc.eval()
    with torch.no_grad():
        out = disc(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 1)
    print("  discriminator: ok")


def test_spatial_locality():
    """Flipping one bit changes a local region, not the whole image."""
    for msg_len in (256, 496):
        enc = Encoder(msg_len, hidden=64)
        enc.eval()
        cover = torch.rand(1, 3, 128, 128)
        msg = (torch.rand(1, msg_len) > 0.5).float()
        with torch.no_grad():
            stego = enc(cover, msg)
            flipped = msg.clone()
            flipped[:, 0] = 1.0 - flipped[:, 0]
            d = (enc(cover, flipped) - stego).abs().mean(dim=(0, 1))
        assert d.max() > 5 * d.mean(), (
            f"msg_len={msg_len}: not localised (ratio {d.max()/d.mean():.1f})")
    print("  spatial locality: ok")


def test_forward_backward():
    """Full forward + backward pass with loss computation, gradients finite."""
    msg_len = 496
    enc, dec, disc = build_models(msg_len, hidden=64)
    noise = NoisePool(enabled=True)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    cover = torch.rand(2, 3, 128, 128)
    msg = random_messages(2, msg_len)

    stego = enc(cover, msg)
    logits = dec(noise(stego, cover))
    loss_msg = bce(logits, msg)
    loss_img = mse(stego, cover)
    loss = loss_msg + 0.7 * loss_img
    loss.backward()

    for name, p in list(enc.named_parameters()) + list(dec.named_parameters()):
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    print("  forward/backward: ok")


def test_optimizer_step():
    """Optimizer step completes without error."""
    msg_len = 496
    enc, dec, _ = build_models(msg_len, hidden=64)
    noise = NoisePool(enabled=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    cover = torch.rand(2, 3, 128, 128)
    msg = random_messages(2, msg_len)

    stego = enc(cover, msg)
    logits = dec(noise(stego, cover))
    loss = bce(logits, msg) + 0.7 * mse(stego, cover)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("  optimizer step: ok")


def test_checkpoint_save_load():
    """Save and reload a 496-bit checkpoint; architecture rebuilds correctly."""
    msg_len = 496
    enc, dec, _ = build_models(msg_len, hidden=64, residual_scale=0.1)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "config": {"msg_len": msg_len, "hidden": 64,
                       "residual_scale": 0.1, "size": 128},
        }, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        assert cfg["msg_len"] == 496

        enc2 = Encoder(cfg["msg_len"], hidden=cfg["hidden"],
                       residual_scale=cfg["residual_scale"])
        dec2 = Decoder(cfg["msg_len"], hidden=cfg["hidden"])
        enc2.load_state_dict(ckpt["encoder"])
        dec2.load_state_dict(ckpt["decoder"])
        enc2.eval(); dec2.eval()

        cover = torch.rand(1, 3, 128, 128)
        msg = (torch.rand(1, msg_len) > 0.5).float()
        with torch.no_grad():
            stego = enc2(cover, msg)
            logits = dec2(stego)
        assert stego.shape == cover.shape
        assert logits.shape == (1, msg_len)
    finally:
        os.unlink(path)
    print("  checkpoint save/load: ok")


def test_old_checkpoint_config():
    """Old 256-bit config still builds the correct architecture."""
    cfg = {"msg_len": 256, "hidden": 64, "residual_scale": 0.1, "size": 128}
    enc = Encoder(cfg["msg_len"], hidden=cfg["hidden"],
                  residual_scale=cfg["residual_scale"])
    dec = Decoder(cfg["msg_len"], hidden=cfg["hidden"])
    assert enc.grid == 16
    assert dec.grid == 16
    print("  old config compat: ok")


def test_payload_capacities():
    """Payload encoding/decoding at all target capacities."""
    cases = [
        # (data_bytes, msg_len, repeat)
        (1,  496, 1),
        (10, 496, 1),
        (30, 496, 1),
        (50, 496, 1),
        (60, 496, 1),   # maximum
        (0,  496, 1),   # empty
        (29, 496, 2),   # repeat=2: (496//2 - 16) // 8 = 29 bytes
        (30, 256, 1),   # old capacity
    ]
    for n_bytes, ml, rep in cases:
        data = bytes(range(n_bytes)) if n_bytes else b""
        bits = encode_payload(data, ml, repeat=rep)
        assert bits.size == ml
        recovered = decode_payload(bits, repeat=rep)
        assert recovered == data, f"{n_bytes}B ml={ml} r={rep}: {recovered!r} != {data!r}"

    # oversize must raise
    for n_bytes, ml, rep in [(61, 496, 1), (31, 256, 1), (30, 496, 2)]:
        try:
            encode_payload(b"X" * n_bytes, ml, repeat=rep)
            assert False, f"should raise: {n_bytes}B ml={ml} r={rep}"
        except ValueError:
            pass

    print("  payload capacities: ok")


def test_payload_utf8():
    """UTF-8 text round-trips through 496-bit payload."""
    text = "Hello 世界! 🌍 café résumé"
    data = text.encode("utf-8")
    assert len(data) <= 60, f"test text too long: {len(data)} bytes"
    bits = encode_payload(data, 496, repeat=1)
    recovered = decode_payload(bits, repeat=1)
    assert recovered == data
    assert recovered.decode("utf-8") == text
    print("  payload utf-8: ok")


def test_payload_key_permutation():
    """Key permutation round-trips on 496-bit vectors."""
    data = b"sixty bytes of data for testing key-based permutation here!!"
    assert len(data) == 60
    bits = encode_payload(data, 496, repeat=1)
    keyed = permute_bits(bits, "testkey")
    # wrong key produces different bits
    assert not np.array_equal(bits, keyed)
    # right key recovers
    unkeyed = unpermute_bits(keyed, "testkey")
    assert np.allclose(bits, unkeyed)
    recovered = decode_payload(unkeyed, repeat=1)
    assert recovered == data
    # wrong key -> garbage
    wrong = decode_payload(unpermute_bits(keyed, "wrongkey"), repeat=1)
    assert wrong != data
    print("  key permutation: ok")


def test_tiling_496():
    """Tiled encode/decode works with 496-bit models."""
    msg_len = 496
    enc, dec, _ = build_models(msg_len, hidden=64)
    enc.eval(); dec.eval()
    msg = (torch.rand(1, msg_len) > 0.5).float()
    for h, w in [(128, 128), (256, 256), (300, 200)]:
        big = torch.rand(1, 3, h, w)
        s = tiled_encode(enc, big, msg, 128)
        assert s.shape == big.shape, f"{h}x{w}: {s.shape}"
        probs = tiled_decode(dec, s, 128)
        assert probs.shape == (msg_len,), f"{h}x{w}: {probs.shape}"
    print("  tiling 496: ok")


def test_ecc_capacity_docs():
    """Document max payload for each ECC setting at 496 bits."""
    ml = 496
    table = []
    for rep in (1, 2, 3, 4):
        cap_bits = (ml // rep) - HEADER_BITS
        cap_bytes = cap_bits // 8 if cap_bits >= 8 else 0
        table.append((rep, cap_bytes))
    expected = [(1, 60), (2, 29), (3, 18), (4, 13)]
    assert table == expected, f"{table} != {expected}"
    print("  ECC capacity table: ok")
    for rep, cap in table:
        print(f"    repeat={rep}: {cap} bytes")


def test_training_smoke():
    """Minimal 2-step training loop on CPU -- verifies no dimension crashes."""
    msg_len = 496
    enc, dec, disc = build_models(msg_len, hidden=32, residual_scale=0.1)
    noise = NoisePool(enabled=True)
    ds = SyntheticImages(n=4, size=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, drop_last=True)

    opt_ed = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    steps = 0
    for cover in loader:
        msg = random_messages(cover.size(0), msg_len)
        stego = enc(cover, msg)

        # disc step
        d_real = disc(cover)
        d_fake = disc(stego.detach())
        loss_d = bce(d_real, torch.zeros_like(d_real)) + bce(d_fake, torch.ones_like(d_fake))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # enc+dec step
        logits = dec(noise(stego, cover))
        loss_msg = bce(logits, msg)
        loss_img = mse(stego, cover)
        loss_adv = bce(disc(stego), torch.zeros_like(disc(stego)))
        loss = loss_msg + 0.7 * loss_img + 0.01 * loss_adv
        opt_ed.zero_grad(set_to_none=True)
        loss.backward()
        opt_ed.step()

        acc = ((logits.detach() > 0.0).float() == msg).float().mean().item()
        steps += 1
        if steps >= 2:
            break

    # save + reload
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({
            "encoder": enc.state_dict(), "decoder": dec.state_dict(),
            "config": {"msg_len": msg_len, "hidden": 32,
                       "residual_scale": 0.1, "size": 64},
        }, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        enc2 = Encoder(ckpt["config"]["msg_len"], hidden=ckpt["config"]["hidden"],
                       residual_scale=ckpt["config"]["residual_scale"])
        enc2.load_state_dict(ckpt["encoder"])
    finally:
        os.unlink(path)
    print(f"  training smoke: ok ({steps} steps, final acc {acc:.3f})")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    print("=== v3 regression tests (496-bit capacity) ===")
    test_msg_grid()
    test_encoder_shapes()
    test_decoder_shapes()
    test_discriminator_unchanged()
    test_spatial_locality()
    test_forward_backward()
    test_optimizer_step()
    test_checkpoint_save_load()
    test_old_checkpoint_config()
    test_payload_capacities()
    test_payload_utf8()
    test_payload_key_permutation()
    test_tiling_496()
    test_ecc_capacity_docs()
    test_training_smoke()
    print("\n=== all tests passed ===")
