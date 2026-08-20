"""Diagnostic: where does the payload die? Isolates each stage of the pipeline."""
import numpy as np, torch, sys
from models import Encoder, Decoder
from data import load_image, save_image, SyntheticImages
from payload import encode_payload, decode_payload
from crypto_key import permute_bits, unpermute_bits

ckpt = torch.load(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model.pt",
                  map_location="cpu", weights_only=False)
cfg = ckpt["config"]
L = cfg["msg_len"]
enc = Encoder(L, hidden=cfg["hidden"], residual_scale=cfg["residual_scale"])
dec = Decoder(L, hidden=cfg["hidden"])
enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])

ds = SyntheticImages(n=64, size=cfg["size"])
cover_batch = torch.stack([ds[i] for i in range(8)])
cover1 = cover_batch[:1]

def acc(logits, msg):
    return ((torch.sigmoid(logits) > 0.5).float() == msg).float().mean().item()

print(f"msg_len={L} size={cfg['size']} residual_scale={cfg['residual_scale']}\n")

# 1. random 50%-density msg, TRAIN mode (batch stats) — this is what training reported
torch.manual_seed(1)
msg_b = (torch.rand(8, L) > 0.5).float()
enc.train(); dec.train()
with torch.no_grad():
    a = acc(dec(enc(cover_batch, msg_b)), msg_b)
print(f"1. random msg, batch=8, TRAIN mode (BN batch stats)  bit-acc {a:.4f}")

# 2. same, EVAL mode (running stats) — this is what embed/extract does
enc.eval(); dec.eval()
with torch.no_grad():
    a = acc(dec(enc(cover_batch, msg_b)), msg_b)
print(f"2. random msg, batch=8, EVAL  mode (BN running stats) bit-acc {a:.4f}")

# 3. eval mode, batch of 1 (the real inference case)
with torch.no_grad():
    a = acc(dec(enc(cover1, msg_b[:1])), msg_b[:1])
print(f"3. random msg, batch=1, EVAL  mode                    bit-acc {a:.4f}")

# 4. real framed payload (sparse! mostly zero padding) - the actual embed path
text = "hi"
bits = encode_payload(text.encode(), L, repeat=1)
print(f"\n   framed payload bit density: {bits.mean():.3f} "
      f"(training used ~0.500)")
bits_k = permute_bits(bits, "hunter2")
msg1 = torch.from_numpy(bits_k.astype(np.float32)).unsqueeze(0)
with torch.no_grad():
    a = acc(dec(enc(cover1, msg1)), msg1)
print(f"4. FRAMED payload, batch=1, EVAL mode                 bit-acc {a:.4f}")

# 5. add the PNG 8-bit quantisation round trip (what save/load actually does)
with torch.no_grad():
    stego = enc(cover1, msg1)
save_image(stego[0], "/tmp/_diag.png")
stego_q = load_image("/tmp/_diag.png").unsqueeze(0)
print(f"   max |stego - requantised| = {(stego-stego_q).abs().max():.5f}")
with torch.no_grad():
    probs = torch.sigmoid(dec(stego_q))[0].numpy()
a = float(((probs > 0.5) == (bits_k > 0.5)).mean())
print(f"5. FRAMED + PNG round trip                            bit-acc {a:.4f}")

got = decode_payload(unpermute_bits(probs, "hunter2"), repeat=1)
hard = (unpermute_bits(probs, "hunter2") > 0.5).astype(np.uint8)
print(f"\n   header decoded n_bytes = {int(''.join(map(str,hard[:16])), 2)} (want 2)")
print(f"   recovered = {got[:32]!r}   (want b'hi')")

# 6. is the residual even doing anything, or is it all cover?
with torch.no_grad():
    s_a = enc(cover1, torch.zeros(1, L))
    s_b = enc(cover1, torch.ones(1, L))
print(f"\n6. residual magnitude: mean|stego-cover| = {(stego-cover1).abs().mean():.5f}")
print(f"   msg sensitivity: mean|stego(0)-stego(1)| = {(s_a-s_b).abs().mean():.5f}")
print(f"   (if ~0, the encoder ignores the message entirely)")
