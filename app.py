"""
app.py
======
Thin Flask server around the existing pipeline (models.py / payload.py /
crypto_key.py / lsb_analysis.py). It does not reimplement any stego logic --
it just calls the same functions embed.py / extract.py / evaluate.py call,
and returns images / JSON to the frontend in static/ + templates/.

    python app.py --model checkpoints/model.pt --port 5000

If no checkpoint exists yet, the server still starts (so the UI is browsable)
but /api/embed and /api/extract return a 409 telling the caller to train
first -- it will never silently fabricate a result.
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image

from models import Encoder, Decoder, tiled_encode, tiled_decode
from data import load_image
from payload import encode_payload, decode_payload, HEADER_BITS
from crypto_key import permute_bits, unpermute_bits
from lsb_analysis import chi_square_pvalue, lsb_run_length

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB uploads

# --size is an escape hatch here exactly as in embed.py, and 0 (native) is the
# default. It resizes to size x size, so a web form's untrusted value needs a
# ceiling; 4096 is the 4K one. Memory no longer scales with it (see GPU_LOCK).
MAX_SIZE = 4096

STATE = {"model_path": None, "device": None, "cfg": None, "encoder": None, "decoder": None}

# app.run() is threaded, so two browser tabs hitting /api/embed at once would run
# two inferences whose peak allocations SUM. Serialising them costs nothing --
# the GPU executes the kernels one at a time regardless -- and bounds peak VRAM
# to a single request.
# ponytail: one global lock. Split per-device if this ever serves >1 GPU.
GPU_LOCK = threading.Lock()


def load_model(model_path: str, device_str: str):
    if STATE["encoder"] is not None and STATE["model_path"] == model_path:
        return
    device = torch.device(device_str)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = Encoder(cfg["msg_len"], hidden=cfg["hidden"],
                      residual_scale=cfg["residual_scale"]).to(device)
    decoder = Decoder(cfg["msg_len"], hidden=cfg["hidden"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    STATE.update(model_path=model_path, device=device, cfg=cfg,
                encoder=encoder, decoder=decoder)


def require_model():
    if STATE["encoder"] is None:
        return jsonify({
            "error": "no_checkpoint",
            "message": ("No trained model loaded. Run `python train.py` (or "
                        "`--synthetic` for a quick smoke test) to produce "
                        "checkpoints/model.pt, then restart the server with "
                        "--model pointing at it."),
        }), 409
    return None


def tensor_to_png_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def int_arg(name: str, default: int, lo: int, hi: int) -> int:
    """Read an integer form field, range-checked. Raises ValueError to be 400'd."""
    raw = (request.form.get(name) or str(default)).strip()
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number (got {raw!r})")
    if not lo <= val <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi} (got {val})")
    return val


def to_gray_u8(t: torch.Tensor) -> np.ndarray:
    r, g, b = t[0], t[1], t[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return (y.clamp(0, 1) * 255).round().byte().cpu().numpy()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    if STATE["encoder"] is None:
        return jsonify({"ready": False})
    cfg = STATE["cfg"]
    capacity_bytes = (cfg["msg_len"] - HEADER_BITS) // 8
    return jsonify({
        "ready": True,
        "model_path": STATE["model_path"],
        "device": str(STATE["device"]),
        "msg_len": cfg["msg_len"],
        "trained_size": cfg["size"],
        "max_capacity_bytes_repeat1": max(capacity_bytes, 0),
    })


@app.post("/api/embed")
def api_embed():
    err = require_model()
    if err:
        return err

    if "cover" not in request.files:
        return jsonify({"error": "missing_file", "message": "attach a cover image"}), 400
    key = request.form.get("key", "")
    message = request.form.get("message", "")
    try:
        # 0 = keep the cover's native resolution, same as embed.py --size 0
        size = int_arg("size", 0, 0, MAX_SIZE)
    except ValueError as e:
        return jsonify({"error": "bad_param", "message": str(e)}), 400
    if not key:
        return jsonify({"error": "missing_key", "message": "a key is required"}), 400
    if not message:
        return jsonify({"error": "missing_message", "message": "enter a message to hide"}), 400

    cfg = STATE["cfg"]
    device = STATE["device"]
    data = message.encode("utf-8")

    try:
        bits = encode_payload(data, cfg["msg_len"], repeat=1)
    except ValueError as e:
        return jsonify({"error": "capacity", "message": str(e)}), 400

    bits = permute_bits(bits, key)
    msg = torch.from_numpy(bits.astype(np.float32)).unsqueeze(0).to(device)

    # size or None -> 0 keeps the cover's NATIVE resolution, same as embed.py
    cover = load_image(request.files["cover"].stream,
                       size=size or None).unsqueeze(0).to(device)

    with GPU_LOCK:
        t0 = time.perf_counter()
        # THE call embed.py:86 makes. Calling STATE["encoder"](cover, msg)
        # directly instead runs one whole-image forward pass: at 3840x2160 that
        # is ~2 GB per activation and ~8.5 GB live, which OOMs any consumer card.
        # tiled_encode peaks at one 16x128px tile batch (~64 MB) for the same
        # native-resolution output, and keeps every tile in-distribution.
        stego = tiled_encode(STATE["encoder"], cover, msg,
                             cfg.get("size", 128))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mse = torch.mean((stego - cover) ** 2).item()
        # steganalysis proof-of-invisibility, run on this exact pair
        g_cover = to_gray_u8(cover[0])
        g_stego = to_gray_u8(stego[0])

    psnr = 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)
    chi_cover = chi_square_pvalue(g_cover)
    chi_stego = chi_square_pvalue(g_stego)

    png = tensor_to_png_bytes(stego[0])
    import base64
    resp = {
        "psnr_db": round(float(psnr), 2),
        "embed_ms": round(elapsed_ms, 1),
        "bytes_hidden": len(data),
        "chi_square_cover": round(chi_cover, 4),
        "chi_square_stego": round(chi_stego, 4),
        "stego_png_b64": base64.b64encode(png).decode("ascii"),
    }
    return jsonify(resp)


@app.post("/api/extract")
def api_extract():
    err = require_model()
    if err:
        return err

    if "stego" not in request.files:
        return jsonify({"error": "missing_file", "message": "attach a stego image"}), 400
    key = request.form.get("key", "")
    if not key:
        return jsonify({"error": "missing_key", "message": "a key is required"}), 400

    device = STATE["device"]
    # native resolution, matching extract.py's --size 0 default
    stego = load_image(request.files["stego"].stream).unsqueeze(0).to(device)

    with GPU_LOCK:
        # THE call extract.py:59 makes. tiled_decode applies sigmoid itself and
        # soft-averages the vote of every whole tile -- which is what lets a 4K
        # stego decode at all: the encoder wrote one message per 128px tile, so
        # a single whole-image pass reads the wrong scale (and OOMs first).
        probs = tiled_decode(STATE["decoder"], stego,
                             STATE["cfg"].get("size", 128)).cpu().numpy()

    bits = unpermute_bits(probs, key)
    data = decode_payload(bits, repeat=1)

    try:
        text = data.decode("utf-8")
        decodable = True
    except UnicodeDecodeError:
        text = None
        decodable = False

    return jsonify({
        "decodable": decodable,
        "text": text,
        "byte_length": len(data),
        "raw_hex": data[:64].hex(),
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/model_v2.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.model):
        load_model(args.model, args.device)
        print(f"loaded checkpoint -> {args.model}")
    else:
        print(f"no checkpoint at {args.model} -- server starting anyway; "
              f"/api/embed and /api/extract will 409 until you train one.")
    app.run(host=args.host, port=args.port, debug=args.debug)
