"""Build a synthetic 'archival tape' with known ground truth from public test clips.

The clips come from two MIT/Apache-licensed repos (real music and real speech):
    git clone --depth 1 https://github.com/tyiannak/pyAudioAnalysis  samples/pyAA
    git clone --depth 1 https://github.com/ina-foss/inaSpeechSegmenter samples/isS

    python eval/make_tape.py samples /tmp/tape.wav
    -> /tmp/tape.wav + /tmp/tape.truth.json (tone, hiss, 4 songs incl. a gapless
       pair, applause, speech, static), then score it with eval_audio.py.
"""
import json, subprocess, sys, numpy as np
SR = 16000
S = (sys.argv[1].rstrip("/") + "/") if len(sys.argv) > 1 else "samples/"
D = S + "pyAA/pyAudioAnalysis/data/"
rng = np.random.default_rng(7)

def load(path, ss=0, t=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(ss), "-i", path]
    if t: cmd += ["-t", str(t)]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.check_output(cmd), dtype=np.float32).copy()

def db(x, level):  # scale to an RMS level in dBFS
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 10 ** (level / 20)

def tone(sec, hz=1000, level=-14):
    t = np.arange(int(sec * SR)) / SR
    return db(np.sin(2 * np.pi * hz * t).astype(np.float32), level)

def noise(sec, level):
    return db(rng.standard_normal(int(sec * SR)).astype(np.float32), level)

def applause(sec, level=-20):
    n = int(sec * SR); y = np.zeros(n, np.float32)
    for _ in range(int(sec * 40)):
        p = rng.integers(0, n - 400); L = rng.integers(80, 400)
        y[p:p + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * rng.uniform(0.3, 1)
    return db(y, level)

parts = [
    ("tone", tone(8)),
    ("hiss", noise(4, -52)),
    ("song", db(load(D + "scottish.wav", 24, 76), -20)),
    ("hiss", noise(2.5, -52)),
    ("song", db(load(D + "recording3.wav"), -22)),
    ("applause", applause(6)),
    ("speech", db(load(D + "diarizationExample.wav"), -22)),
    ("hiss", noise(1.5, -52)),
    ("song", db(load(D + "beat/100 BPM - Rhythm patterns - Salsa.mp3", 0, 60), -20)),
    ("song", db(load(D + "beat/200 BPM (goa psy trance).mp3", 0, 60), -20)),
    ("hiss", noise(8, -52)),
    ("static", noise(10, -22)),
    ("speech", db(load(S + "isS/media/lamartine.wav"), -22)),
    ("hiss", noise(7, -52)),
]
y = np.concatenate([p for _, p in parts])
y = y + noise(len(y) / SR, -55)[: len(y)]          # tape hiss floor under everything
truth, t = [], 0.0
for kind, p in parts:
    truth.append({"start": round(t, 2), "end": round(t + len(p) / SR, 2), "kind": kind}); t += len(p) / SR
out = sys.argv[2] if len(sys.argv) > 2 else "tape.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                "-c:a", "pcm_s16le", out], input=np.clip(y, -1, 1).astype(np.float32).tobytes(), check=True)
json.dump(truth, open(out.rsplit(".", 1)[0] + ".truth.json", "w"), indent=1)
for r in truth: print(r)
