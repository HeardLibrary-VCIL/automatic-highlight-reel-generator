"""Stage B of the audio-fusion pipeline: join transcript turns onto shots.

Takes the shots from segment_shots.detect_shots() and the speaker turns from
transcribe.py's transcript.json, and attaches to each shot the words spoken
during it plus which speakers were talking. This is the fusion "join": after it,
every shot carries BOTH what it looks like (a frame, added in Stage C) and what
was said over it (text + speaker) -- which is what lets a B-roll cutaway inherit
the parent story's continuous narration instead of being read as its own topic.

Output shots_fused.json = [{start, end, speakers, speech_frac, text}, ...].
No API, no model -- pure interval overlap. Stage C classifies these; Stage D
merges them into stories using label + speaker/transcript continuity.

Usage:
    python segment_fuse.py video.mp4 transcript.json -o shots_fused.json
    python segment_fuse.py video.mp4 transcript.json --window 31.87 1739.0   # skip re-trim
"""

import argparse
import json
import sys

from analyze_deadspace import get_duration, analyze
from segment_shots import detect_shots


def load_turns(path) -> list:
    turns = json.load(open(path))
    return [(float(t["start"]), float(t["end"]), t.get("speaker", "spk_?"), t.get("text", ""))
            for t in turns]


def attach(shots, turns) -> list:
    """For each shot, gather overlapping speaker turns.

    A turn counts for a shot if their time spans overlap at all; the shot's
    `speech_frac` is the fraction of its duration covered by speech (low frac =>
    music/dance/silent card, where Stage C should fall back to vision). `text` is
    the concatenation of every overlapping turn (kept whole for readability, so a
    turn that spans a cut shows on both shots -- intentional: that continuity is
    the signal that the two shots belong together)."""
    out = []
    for ss, ee in shots:
        dur = ee - ss
        speakers, texts, covered = [], [], 0.0
        for ts, te, spk, txt in turns:
            ov = min(te, ee) - max(ts, ss)
            if ov > 0:
                covered += ov
                if spk not in speakers:
                    speakers.append(spk)
                texts.append((ts, spk, txt))
        texts.sort()
        out.append({
            "start": round(ss, 2), "end": round(ee, 2),
            "speakers": speakers,
            "speech_frac": round(min(covered / dur, 1.0), 2) if dur else 0.0,
            "text": " ".join(t for _, _, t in texts).strip(),
        })
    return out


def merge_by_speaker(shots, host="spk_0", min_speech=0.2) -> list:
    """Collapse fused shots into story segments using speaker continuity alone
    (no LLM). Adjacent shots merge when they share the same signature:

      - host-only speech        -> "host" (intro/outro, or narration over B-roll)
      - a non-host participant  -> "guest" (an interview/segment ABOUT that person)
      - <min_speech speech       -> "nospeech" (music/dance/silent card) -- absorbed
                                    into whichever neighbour it sits inside

    Grouping on the NON-host participant (not on whether the host is audible in a
    given shot) is what stops an interview flickering between host-question and
    guest-answer shots. This already recovers most of the true structure on
    RCC_183 (115 shots -> ~25 segments); Stage C only has to NAME each one."""
    def sig(s):
        if s["speech_frac"] < min_speech:
            return ("nospeech",)
        guests = tuple(x for x in s["speakers"] if x != host)
        return ("guest",) + guests if guests else ("host",)

    segs = []
    for s in shots:
        k = sig(s)
        if segs and (k == segs[-1]["sig"] or k == ("nospeech",)):
            segs[-1]["end"] = s["end"]
        elif segs and segs[-1]["sig"] == ("nospeech",):   # pure-nospeech run adopts this shot's sig
            segs[-1]["sig"], segs[-1]["end"] = k, s["end"]
        else:
            segs.append({"sig": k, "start": s["start"], "end": s["end"]})
        segs[-1].setdefault("shots", []).append(s)

    out = []
    for g in segs:
        speakers = []
        for s in g["shots"]:
            for spk in s["speakers"]:
                if spk not in speakers:
                    speakers.append(spk)
        role = {"host": "host", "nospeech": "nospeech"}.get(g["sig"][0], "guest")
        out.append({"start": g["start"], "end": g["end"], "role": role, "speakers": speakers})
    return out


def main():
    p = argparse.ArgumentParser(description="Join transcript turns onto shots (Stage B).")
    p.add_argument("input_video")
    p.add_argument("transcript", help="transcript.json from transcribe.py")
    p.add_argument("-o", "--output", default="shots_fused.json")
    p.add_argument("--detector", choices=["adaptive", "content"], default="adaptive")
    p.add_argument("--min-shot", type=float, default=1.5)
    p.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                   help="Content window to skip re-running the dead-space trim.")
    args = p.parse_args()

    duration = get_duration(args.input_video)
    if args.window:
        window = tuple(args.window)
    else:
        prop = analyze(args.input_video)
        window = ((prop.content_start, prop.content_end)
                  if prop.status == "OK" and prop.kept > 0 else (0.0, duration))

    shots = detect_shots(args.input_video, window, args.detector, args.min_shot)
    turns = load_turns(args.transcript)
    fused = attach(shots, turns)

    print(f"Content window: {window[0]:.1f}->{window[1]:.1f}s   shots={len(shots)}   "
          f"turns={len(turns)}", file=sys.stderr)
    for f in fused:
        spk = ",".join(f["speakers"]) or "-"
        print(f"  [{f['start']:7.1f}->{f['end']:7.1f}] sp={f['speech_frac']:.2f} "
              f"{spk:14s} {f['text'][:64]}")

    with open(args.output, "w") as fh:
        json.dump({"video": args.input_video, "window": [round(window[0], 2), round(window[1], 2)],
                   "shots": fused}, fh, indent=2)
    print(f"\nWrote {len(fused)} fused shots to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
