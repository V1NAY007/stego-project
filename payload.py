"""
payload.py
==========
Convert human payloads (UTF-8 text or raw bytes) <-> fixed-length bit vectors,
with an optional repetition error-correcting code (ECC) for robustness.

The network embeds a *fixed* number of bits per image (the model's `msg_len`).
We frame the real payload as:

    [ 16-bit length header | payload bits | zero padding ]   (before ECC)

then optionally repeat every bit `repeat` times and majority-vote on decode.
Repetition is the simplest ECC; swap in BCH / LDPC later if you want more.
"""

from __future__ import annotations

import numpy as np

HEADER_BITS = 16  # up to 65535 payload bytes describable


def bytes_to_bits(data: bytes) -> np.ndarray:
    """bytes -> 1-D uint8 array of bits (MSB first)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """1-D bit array (len multiple of 8) -> bytes."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8 != 0:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def _int_to_bits(value: int, n: int) -> np.ndarray:
    return np.array([(value >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def encode_payload(data: bytes, msg_len: int, repeat: int = 1) -> np.ndarray:
    """
    Frame `data` into exactly `msg_len` bits (post-ECC).

    Raises ValueError if the payload does not fit.
    """
    payload_bits = bytes_to_bits(data)
    n_bytes = len(data)
    if n_bytes >= (1 << HEADER_BITS):
        raise ValueError("payload too large for 16-bit length header")

    header = _int_to_bits(n_bytes, HEADER_BITS)
    framed = np.concatenate([header, payload_bits])

    capacity = msg_len // repeat
    if framed.size > capacity:
        raise ValueError(
            f"payload needs {framed.size} bits but capacity is {capacity} "
            f"(msg_len={msg_len}, repeat={repeat}). "
            f"Use a larger model msg_len, a smaller payload, or repeat=1."
        )

    padded = np.zeros(capacity, dtype=np.uint8)
    padded[: framed.size] = framed

    if repeat > 1:
        padded = np.repeat(padded, repeat)

    # pad/truncate to EXACTLY msg_len so it matches the encoder's input width
    out = np.zeros(msg_len, dtype=np.uint8)
    k = min(padded.size, msg_len)
    out[:k] = padded[:k]
    return out.astype(np.float32)


def decode_payload(bits: np.ndarray, repeat: int = 1) -> bytes:
    """
    Inverse of encode_payload. `bits` may be soft (probabilities); we threshold.
    """
    soft = np.asarray(bits, dtype=np.float64)

    if repeat > 1:
        usable = (soft.size // repeat) * repeat
        # Average the PROBABILITIES, then threshold once (soft-decision decoding).
        # Thresholding to hard bits first discarded the decoder's confidence:
        # three copies at 0.49/0.49/0.99 are a 2-1 hard vote for 0, but they
        # average to 0.66, which is the better call. It also removes the tie bias
        # at even `repeat`, where 1-vs-1 used to resolve to 0 because the old
        # majority test was `mean > 0.5`.
        soft = soft[:usable].reshape(-1, repeat).mean(axis=1)

    hard = (soft > 0.5).astype(np.uint8)

    header = hard[:HEADER_BITS]
    n_bytes = _bits_to_int(header)
    start = HEADER_BITS
    end = start + n_bytes * 8
    if end > hard.size:
        end = hard.size  # be forgiving on corrupted headers
    return bits_to_bytes(hard[start:end])


if __name__ == "__main__":
    # msg_len must be >= (16-bit header + payload*8) * repeat
    for text, rep, msg_len in [
        ("hi", 1, 64),
        ("secret meeting @ 9", 4, 1024),   # 18 bytes -> 160 bits * 4 = 640 <= 1024
        ("", 2, 64),
        ("The quick brown fox.", 1, 256),
    ]:
        enc = encode_payload(text.encode(), msg_len, repeat=rep)
        assert enc.size == msg_len
        dec = decode_payload(enc, repeat=rep).decode()
        assert dec == text, (text, dec)
    print("payload self-check passed")
