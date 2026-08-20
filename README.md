# Neural (non-LSB) Steganography

A CNN-based steganography system that hides a message inside an image **without
touching bits directly**. Instead of flipping least-significant bits (LSB), a
convolutional **encoder** learns to add a faint, image-shaped perturbation
spread across the whole picture and across *all* bit planes. A paired
convolutional **decoder** — plus a secret **key** — is the only thing that can
read the payload back.

This is the HiDDeN / SteganoGAN family of "deep steganography." It is a
legitimate, published research area (watermarking, robust data hiding, privacy
research). Use it responsibly and legally.

## Why this evades LSB steganalysis

Classic LSB embedding randomises the least-significant bit of many pixels. That
leaves a statistical fingerprint: the histogram counts of each value pair
`(2k, 2k+1)` get *equalised*. The **chi-square pairs-of-values attack**
(Westfeld–Pfitzmann) detects exactly this.

This system never does that. The encoder emits a small continuous residual
(`stego = cover + tanh(residual) * scale`) that is then quantised to 8-bit. The
signal lands across every bit plane and only mildly reshapes the histogram, so
the chi-square attack — and RS / sample-pair style detectors built on the same
LSB assumption — stay quiet.

Demonstrated in `lsb_analysis.py`:

| image                | chi-square p(embedded) | verdict     |
|----------------------|------------------------|-------------|
| clean cover          | ~0.00                  | looks clean |
| classic LSB stego    | ~0.97                  | SUSPICIOUS  |
| **this CNN stego**   | stays ~0.00            | looks clean |

## Why "only we" can read it

Two independent secrets:

1. **The trained decoder weights.** Without your exact `model.pt`, the payload
   is unrecoverable. This is the primary secret.
2. **The key** (`crypto_key.py`). A string key seeds a deterministic
   permutation of the message bits (and an optional conditioning vector). Even
   someone with a similar decoder gets scrambled bits without the key. Wrong
   key → garbage.

## Files

| file              | what it does                                             |
|-------------------|---------------------------------------------------------|
| `models.py`       | Encoder / Decoder / Discriminator CNNs                  |
| `noise_layers.py` | Differentiable corruptions for robustness training      |
| `crypto_key.py`   | Key → bit permutation + conditioning (the key layer)    |
| `payload.py`      | text/bytes ↔ fixed-length bits, with repetition ECC     |
| `data.py`         | image loading, message sampling, synthetic images       |
| `train.py`        | joint training loop                                     |
| `embed.py`        | CLI: hide a message in a cover image                    |
| `extract.py`      | CLI: recover a message from a stego image               |
| `lsb_analysis.py` | chi-square LSB detector — shows the contrast            |
| `selftest.py`     | fast end-to-end wiring test on synthetic data           |

## Install

```bash
pip install -r requirements.txt   # torch, numpy, pillow, scipy
```

## Quick start

**0. Sanity-check the pipeline (no dataset, ~1–2 min CPU):**
```bash
python selftest.py
```

**1. Train.** Point it at a folder of images (more + bigger = better):
```bash
python train.py --data /path/to/images --size 128 --msg-len 256 --epochs 40
```
No dataset handy? Train on synthetic images just to see it work:
```bash
python train.py --synthetic --size 64 --msg-len 64 --epochs 5 --batch 8
```
Output: `checkpoints/model.pt`.

**2. Embed a message:**
```bash
python embed.py --model checkpoints/model.pt \
    --cover cover.png --out stego.png \
    --message "meet at the docks, midnight" --key "hunter2" --repeat 4
```

**3. Extract it (correct key required):**
```bash
python extract.py --model checkpoints/model.pt \
    --stego stego.png --key "hunter2" --repeat 4
```

**4. Confirm it hides from LSB detectors:**
```bash
python lsb_analysis.py --make-demo cover.png        # a good demo cover
python lsb_analysis.py --image cover.png --make-lsb # clean vs classic LSB
python lsb_analysis.py --image stego.png            # your CNN stego -> clean
```

## Important operational notes

- **Always distribute stego images as PNG.** JPEG re-compression will likely
  destroy the payload unless you add a differentiable-JPEG noise layer to
  `noise_layers.py` and train with it. `save_image` writes PNG for you.
- **Capacity.** The model carries a fixed `msg_len` bits. Effective text
  capacity ≈ `(msg_len / repeat - 16) / 8` bytes. Raise `--msg-len` for more
  room; raise `--repeat` for more robustness (at the cost of capacity).
- **Quality vs invisibility.** `--img-weight` / `--residual-scale` trade payload
  accuracy against imperceptibility (PSNR). Watch both during training.
- **Robustness.** Train with the noise pool (default on) so the message survives
  quantisation, mild blur, and noise. For JPEG/crop robustness, extend the noise
  pool accordingly.

## Extending

- Add **DiffJPEG** to `noise_layers.py` for JPEG-robust payloads.
- Swap repetition ECC in `payload.py` for **BCH/LDPC** for higher reliable capacity.
- Add an **LPIPS** perceptual loss in `train.py` for better invisibility.
- Add a learned **steganalysis critic** (a CNN that tries to detect your stego)
  and train adversarially against it for even stronger undetectability.
