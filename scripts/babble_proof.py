"""Babble proof for pad-aware batching.

Run inside the tts image with this tree at /app and a GPU:
  VOICE=/app/reference_audio/<voice>.wav OUT=/tmp/out python3 scripts/babble_proof.py
A short row batched next to a long row must end about where its sequential
take ends, not where the long row ends. Red on the head before this fix
(short rows ran 4-6 s for a 1 s line), green after.
"""
import os, sys, time
sys.path.insert(0, "/app")
os.chdir("/app")
import torch
import engine

VOICE = os.environ.get("VOICE", "/app/reference_audio/quark.wav")
SHORT = "Make it so."
LONG = ("Rule of Acquisition number two hundred and eighty five: no good deed ever "
        "goes unpunished. You merged the fix for free? Hew-mons. A twelve gigabyte "
        "image, and not one slip of latinum changed hands.")

assert engine.load_model(), "model did not load"
engine.TTS_BATCH_SIZE = 4
sr = engine.chatterbox_model.sr

def dur(w):
    return w.shape[-1] / sr

t = time.time()
seq_short = [dur(engine.synthesize(SHORT, VOICE)[0]) for _ in range(3)]
seq_long = dur(engine.synthesize(LONG, VOICE)[0])
t_seq = time.time() - t

t = time.time()
wavs, _ = engine.synthesize_batch([SHORT, LONG, SHORT, LONG], VOICE)
t_batch = time.time() - t
assert wavs is not None, "batch fell back"
b_short = [dur(wavs[0]), dur(wavs[2])]
b_long = [dur(wavs[1]), dur(wavs[3])]

print(f"seq   short {seq_short} s  long {seq_long:.1f} s   ({t_seq:.1f} s wall for 4 calls)")
print(f"batch short {b_short} s  long {b_long} s   ({t_batch:.1f} s wall for 4 rows)")

out = os.environ.get("OUT", "/tmp/babble_proof")
os.makedirs(out, exist_ok=True)
import wave
def save(path, w):
    pcm = (w.squeeze(0).clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy().tobytes()
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm)
save(f"{out}/short-seq.wav", engine.synthesize(SHORT, VOICE)[0])
save(f"{out}/short-batch.wav", wavs[0])
save(f"{out}/long-batch.wav", wavs[1])

ceiling = 1.6 * max(seq_short)
for d in b_short:
    assert d < ceiling, f"short row babbles: {d:.1f} s batched vs {max(seq_short):.1f} s sequential"
    assert d < 0.6 * min(b_long), f"short row runs to the long row's end: {d:.1f} vs {min(b_long):.1f}"
print("PASS: padded rows stop where their text stops")
