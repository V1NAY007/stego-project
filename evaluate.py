"""
evaluate.py
===========
Held-out evaluation of a trained encoder/decoder, at NATIVE resolution and
through the same tiling that `embed.py` / `extract.py` use -- i.e. the real
path, not the training path.

    python evaluate.py --model checkpoints/model_v2.pt \
        --data datasets/coco2017/val2017 --n 60 --key hunter2

Reports, per repetition factor:
  * capacity in bytes, and EXACT message recovery rate -- the number that
    matters. Bit accuracy is misleading: one wrong bit corrupts a whole byte,
    so 0.999 bit accuracy over 256 bits still fails ~23% of messages.
  * raw bit accuracy, PSNR, and embed/extract wall time.

Plus a steganalysis section comparing chi-square (Westfeld-Pfitzmann) on the
cover, the CNN stego, and classic LSB replacement at BOTH the same payload and
a full fill. The matched-payload arm is the honest comparison and it is
unflattering to the usual demo -- see the note it prints.

Writes examples/ (cover, stego, x20 amplified difference) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import IMG_EXTS, load_image, save_image
from models import Decoder, Encoder, tiled_decode, tiled_encode
from payload import HEADER_BITS, decode_payload, encode_payload
from crypto_key import permute_bits, unpermute_bits
from lsb_analysis import chi_square_pvalue, lsb_embed, lsb_run_length


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="folder of HELD-OUT images")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--key", default="hunter2")
    p.add_argument("--repeats", default="1,2,3,4")
    p.add_argument("--size", type=int, default=0,
                   help="0 = native resolution (what embed.py now does)")
    p.add_argument("--tile", type=int, default=0,
                   help="0 (default) = the model's training size, as embed.py uses")
    p.add_argument("--out-dir", default="eval_out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def psnr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def sync(device: torch.device) -> None:
    """CUDA launches are async, so timing without this measures queueing only."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def to_gray_u8(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float in [0,1] -> (H,W) uint8 luma, matching PIL's 'L'."""
    r, g, b = t[0], t[1], t[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return (y.clamp(0, 1) * 255).round().byte().cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    L = cfg["msg_len"]
    encoder = Encoder(L, hidden=cfg["hidden"],
                      residual_scale=cfg["residual_scale"]).to(device)
    decoder = Decoder(L, hidden=cfg["hidden"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval(); decoder.eval()

    paths = sorted(p for p in Path(args.data).rglob("*")
                   if p.suffix.lower() in IMG_EXTS)[: args.n]
    if not paths:
        raise SystemExit(f"no images found in {args.data}")
    repeats = [int(r) for r in args.repeats.split(",")]
    tile = args.tile or cfg.get("size", 128)
    tmp = os.path.join(args.out_dir, "_roundtrip.png")

    print(f"model: msg_len={L} trained at {cfg['size']}px | "
          f"{len(paths)} held-out images | "
          f"{'native resolution' if not args.size else f'{args.size}px'} | "
          f"{tile}px tiles")

    # warm up cuDNN autotuning and lazy CUDA init, or the first timed image
    # absorbs all of it and reads ~20x slow
    warm = load_image(str(paths[0]), size=args.size or None).unsqueeze(0).to(device)
    tiled_decode(decoder, tiled_encode(encoder, warm,
                                       torch.zeros(1, L, device=device), tile), tile)
    sync(device)
    del warm

    rows = []
    for rep in repeats:
        cap_bits = (L // rep) - HEADER_BITS
        if cap_bits < 8:
            print(f"repeat {rep}: capacity {cap_bits} bits, too small -- skipped")
            continue
        n_bytes = cap_bits // 8

        exact = wrongkey_exact = 0
        accs, psnrs, t_emb, t_ext = [], [], [], []
        for i, path in enumerate(paths):
            # deterministic per-image payload that FILLS capacity (the stress
            # case -- no zero padding to make the decoder's job easier)
            rng = np.random.RandomState(i)
            text = bytes(rng.randint(32, 127, n_bytes).astype(np.uint8))

            bits = encode_payload(text, L, repeat=rep)
            keyed = permute_bits(bits, args.key)
            msg = torch.from_numpy(keyed.astype(np.float32)).unsqueeze(0).to(device)
            cover = load_image(str(path), size=args.size or None)
            cover = cover.unsqueeze(0).to(device)

            t0 = time.perf_counter()
            stego = tiled_encode(encoder, cover, msg, tile)
            sync(device)
            t_emb.append(time.perf_counter() - t0)

            # go through a real PNG, exactly as a user would
            save_image(stego[0], tmp)
            reloaded = load_image(tmp).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            probs = tiled_decode(decoder, reloaded, tile).cpu().numpy()
            sync(device)
            t_ext.append(time.perf_counter() - t0)

            accs.append(float(((probs > 0.5) == (keyed > 0.5)).mean()))
            psnrs.append(psnr_db(stego, cover))
            exact += decode_payload(unpermute_bits(probs, args.key), rep) == text
            wrongkey_exact += (
                decode_payload(unpermute_bits(probs, args.key + "X"), rep) == text)

            if i == 0 and rep == repeats[0]:
                save_image(cover[0], os.path.join(args.out_dir, "example_cover.png"))
                save_image(stego[0], os.path.join(args.out_dir, "example_stego.png"))
                save_image(((stego[0] - cover[0]).abs() * 20).clamp(0, 1),
                           os.path.join(args.out_dir, "example_diff_x20.png"))

        rows.append({
            "repeat": rep, "payload_bytes": n_bytes,
            "exact_recovery": exact / len(paths),
            "bit_acc": float(np.mean(accs)), "psnr_db": float(np.mean(psnrs)),
            "embed_ms": float(np.mean(t_emb) * 1000),
            "extract_ms": float(np.mean(t_ext) * 1000),
            "wrong_key_recovery": wrongkey_exact / len(paths),
        })
        r = rows[-1]
        print(f"  repeat {rep}: {n_bytes:2d} B | exact {r['exact_recovery']:6.1%} | "
              f"bit-acc {r['bit_acc']:.4f} | PSNR {r['psnr_db']:5.2f} dB | "
              f"embed {r['embed_ms']:6.1f} ms | extract {r['extract_ms']:6.1f} ms")

    print(f"\n{'repeat':>6} {'bytes':>6} {'exact recovery':>15} {'bit-acc':>8} "
          f"{'PSNR dB':>8} {'wrong key':>10}")
    for r in rows:
        print(f"{r['repeat']:>6} {r['payload_bytes']:>6} "
              f"{r['exact_recovery']:>14.1%} {r['bit_acc']:>8.4f} "
              f"{r['psnr_db']:>8.2f} {r['wrong_key_recovery']:>9.1%}")

    # ---- steganalysis: chi-square on cover vs CNN stego vs classic LSB ----
    rep0 = repeats[0]
    cap0 = (L // rep0) - HEADER_BITS
    chi = {"cover": [], "cnn_stego": [], "lsb_matched": [], "lsb_full": []}
    runlen = {"cover": [], "cnn_stego": [], "lsb_full": []}
    for i, path in enumerate(paths):
        rng = np.random.RandomState(i)
        text = bytes(rng.randint(32, 127, cap0 // 8).astype(np.uint8))
        bits = permute_bits(encode_payload(text, L, repeat=rep0), args.key)
        msg = torch.from_numpy(bits.astype(np.float32)).unsqueeze(0).to(device)
        cover = load_image(str(path), size=args.size or None).unsqueeze(0).to(device)
        stego = tiled_encode(encoder, cover, msg, tile)

        g_cover = to_gray_u8(cover[0])
        g_stego = to_gray_u8(stego[0].clamp(0, 1))
        chi["cover"].append(chi_square_pvalue(g_cover))
        chi["cnn_stego"].append(chi_square_pvalue(g_stego))
        # same number of payload bits as the CNN carried, via classic LSB
        chi["lsb_matched"].append(
            chi_square_pvalue(lsb_embed(g_cover, fill=cap0 / g_cover.size, seed=i)))
        chi["lsb_full"].append(chi_square_pvalue(lsb_embed(g_cover, fill=1.0, seed=i)))
        runlen["cover"].append(lsb_run_length(g_cover))
        runlen["cnn_stego"].append(lsb_run_length(g_stego))
        runlen["lsb_full"].append(lsb_run_length(lsb_embed(g_cover, 1.0, seed=i)))

    print(f"\nchi-square p(embedded), mean over {len(paths)} images "
          f"(>0.5 => flagged as LSB stego)")
    for k in ("cover", "cnn_stego", "lsb_matched", "lsb_full"):
        flagged = float(np.mean([p > 0.5 for p in chi[k]]))
        print(f"  {k:12s} p={np.mean(chi[k]):.4f}   flagged {flagged:6.1%}")
    print("  LSB mean run length: "
          + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in runlen.items()))
    print(f"\nNote: lsb_matched carries the SAME {cap0} payload bits as the CNN. At "
          f"this payload\n  chi-square detects neither, so 'invisible to "
          f"chi-square' is only a real claim\n  against lsb_full. Report the "
          f"matched arm too -- the CNN's edge here is that it\n  survives "
          f"re-quantisation, not that it beats LSB on this particular detector.")

    summary = {"model": args.model, "config": cfg, "n_images": len(paths),
               "native_resolution": not args.size, "tile": tile, "rows": rows,
               "chi_square_mean": {k: float(np.mean(v)) for k, v in chi.items()},
               "lsb_run_length_mean": {k: float(np.mean(v)) for k, v in runlen.items()}}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    os.remove(tmp)
    print(f"\nwrote {args.out_dir}/summary.json and example_{{cover,stego,diff_x20}}.png")


if __name__ == "__main__":
    main()
