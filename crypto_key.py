"""
crypto_key.py
=============
The "only we can read it" layer.

Deep steganography already gives you secrecy for free: without the *exact trained
decoder weights*, an attacker cannot recover the payload. But we layer a second,
cheap secret on top so that even someone who somehow obtained a similar decoder
still gets garbage without the key:

    key (string)  ->  SHA-256  ->  seed  ->  deterministic bit permutation
                                          ->  deterministic conditioning vector

The encoder embeds the *permuted* bits (optionally conditioned on the key vector);
the decoder recovers them and applies the *inverse* permutation. Wrong key ->
scrambled bits -> noise.

This module is pure NumPy so it can be unit-tested without a GPU / PyTorch.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _seed_from_key(key: str, salt: str = "") -> int:
    """Turn an arbitrary string key into a 32-bit integer seed.

    numpy's legacy RandomState requires a seed in [0, 2**32 - 1]. We fold the
    SHA-256 digest down to 32 bits; the full digest still influences the result.
    """
    h = hashlib.sha256((salt + "::" + key).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")  # 32-bit seed


def key_permutation(key: str, length: int) -> np.ndarray:
    """
    Deterministic permutation of indices [0, length) derived from `key`.
    Same key + length always yields the same permutation.
    """
    rng = np.random.RandomState(_seed_from_key(key, "perm"))
    return rng.permutation(length)


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Return the inverse of a permutation array."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return inv


def key_conditioning(key: str, dim: int) -> np.ndarray:
    """
    A fixed pseudo-random vector in [-1, 1] derived from the key.
    Optionally fed into the encoder/decoder as an extra conditioning signal so
    the network's behaviour itself depends on the key, not just the bit order.
    """
    rng = np.random.RandomState(_seed_from_key(key, "cond"))
    return (rng.rand(dim).astype(np.float32) * 2.0) - 1.0


def permute_bits(bits: np.ndarray, key: str) -> np.ndarray:
    """Apply the key permutation to a 1-D bit array."""
    perm = key_permutation(key, bits.shape[-1])
    return bits[..., perm]


def unpermute_bits(bits: np.ndarray, key: str) -> np.ndarray:
    """Undo the key permutation."""
    perm = key_permutation(key, bits.shape[-1])
    inv = inverse_permutation(perm)
    return bits[..., inv]


if __name__ == "__main__":
    # tiny self-check
    b = np.arange(30) % 2
    for k in ["hunter2", "correct horse battery staple"]:
        p = permute_bits(b, k)
        r = unpermute_bits(p, k)
        assert np.array_equal(b, r), "round trip failed"
        # wrong key should (almost surely) scramble
        wrong = unpermute_bits(p, k + "!")
        assert not np.array_equal(b, wrong)
    print("crypto_key self-check passed")
