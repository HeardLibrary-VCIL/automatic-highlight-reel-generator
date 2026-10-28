"""Score audio_segment against a ground-truth JSON ([{start,end,kind}]).

Reports row-level label accuracy and boundary precision/recall within a
tolerance. `--oracle-speech` feeds the truth's speech spans in as if they were
Amazon Transcribe turns, which is what the deployed pipeline has available.

    python eval/make_tape.py samples /tmp/tape.wav     # build a labelled test tape
    python eval/eval_audio.py /tmp/tape.wav /tmp/tape.truth.json [--oracle-speech] [--tol 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio_segment as A  # noqa: E402

# Ground-truth kinds -> the label the segmenter should produce.
TRUTH_MAP = {"song": "song", "music": "song", "speech": "speech", "male": "speech",
             "female": "speech", "applause": "applause", "tone": "tone", "static": "static",
             "hiss": "dead", "noEnergy": "dead", "silence": "dead", "noise": "dead"}
DEAD = {"silence", "hiss", "tone", "static", "dead"}


def simplify_truth(truth, min_dead=A.MIN_DEAD_DUR):
    """Pauses shorter than min_dead are not boundaries (a breath in speech),
    except between two songs, where any gap is the song change."""
    out = []
    for i, s in enumerate(truth):
        s = dict(s)
        short_dead = norm(s["kind"]) == "dead" and s["end"] - s["start"] < min_dead
        prev_song = out and norm(out[-1]["kind"]) == "song"
        next_song = i + 1 < len(truth) and norm(truth[i + 1]["kind"]) == "song"
        if short_dead and out and not (prev_song and next_song):
            out[-1]["end"] = s["end"]
            continue
        if out and norm(out[-1]["kind"]) == norm(s["kind"]) and norm(s["kind"]) != "song":
            out[-1]["end"] = s["end"]
            continue
        out.append(s)
    return out


def label_at(segs, t, key="kind"):
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s[key]
    return None


def norm(k):
    k = TRUTH_MAP.get(k, k)
    return "dead" if k in ("silence", "hiss") else k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("truth")
    p.add_argument("--oracle-speech", action="store_true")
    p.add_argument("--tol", type=float, default=2.0)
    a = p.parse_args()
    truth = simplify_truth(json.load(open(a.truth)))
    turns = ([{"start": s["start"], "end": s["end"]} for s in truth
              if TRUTH_MAP.get(s["kind"]) == "speech"] if a.oracle_speech else None)
    regs, R, _, _ = A.segment_audio(a.audio, turns)
    pred = [{"start": r.start, "end": r.end, "kind": r.kind} for r in regs]

    ts = np.arange(0, R.duration, 0.25)
    ok = [norm(label_at(pred, t) or "") == norm(label_at(truth, t) or "") for t in ts]
    acc = float(np.mean(ok))

    # Boundaries: every change of (normalised) kind, plus song->song cuts.
    def bounds(segs):
        out = []
        for x, y in zip(segs[:-1], segs[1:]):
            if norm(x["kind"]) != norm(y["kind"]) or norm(x["kind"]) == "song":
                out.append(y["start"])
        return out
    tb, pb = bounds(truth), bounds(pred)
    hit_t = [any(abs(b - c) <= a.tol for c in pb) for b in tb]
    hit_p = [any(abs(b - c) <= a.tol for c in tb) for b in pb]
    rec = float(np.mean(hit_t)) if tb else 1.0
    prec = float(np.mean(hit_p)) if pb else 1.0
    print(f"{Path(a.audio).name}: row accuracy {acc:.1%} | boundaries recall {rec:.0%} "
          f"({sum(hit_t)}/{len(tb)}) precision {prec:.0%} ({sum(hit_p)}/{len(pb)}) @ +-{a.tol:g}s"
          + (" [oracle speech]" if a.oracle_speech else ""))
    missed = [b for b, h in zip(tb, hit_t) if not h]
    extra = [b for b, h in zip(pb, hit_p) if not h]
    if missed:
        print("  missed:", missed)
    if extra:
        print("  extra :", extra)
    return acc, rec, prec


if __name__ == "__main__":
    main()
