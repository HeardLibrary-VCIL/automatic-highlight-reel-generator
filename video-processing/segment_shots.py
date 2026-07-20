"""Shot-first content-type segmentation (Level 1 shots + Level 2 classify).

An alternative front end to segment_content.py. Instead of blind-slicing the
program into a fixed 30s grid, it detects REAL shot boundaries with PySceneDetect
and treats each shot as the classification unit, then merges same-label
neighbors. Two things fall out of that:

  - Boundaries land on true cuts, so the A/B bisection refinement that
    segment_content needs to un-snap boundaries from the 30s grid is gone.
  - Each classification window is a homogeneous camera take, not an arbitrary
    30s slice that can straddle a cut (the old grid mislabeled a shot that only
    filled half its window -- e.g. an animation read as 'commercial').

Pipeline:
    analyze_deadspace trim -> PySceneDetect shots -> coalesce micro-shots
    -> sample frames per shot -> classify (closed set) -> confirm -> smooth
    -> merge equal labels -> consolidate.

Everything after "sample" is reused verbatim from segment_content.py; the only
new part is detect_shots() replacing sample_windows(). Title cards land on their
own shots here (PySceneDetect isolates each one), which is what a later OCR pass
would key on to NAME a segment -- not built yet.

Usage:
    python segment_shots.py video.mp4                 # trim + shots + classify
    python segment_shots.py video.mp4 --no-classify   # shots only, no API (free)
    python segment_shots.py video.mp4 --min-shot 2.0  # coalesce sub-2s micro-cuts

Setup: same as segment_content.py -- `pip install anthropic scenedetect` and
`export ANTHROPIC_API_KEY=...` (classify step only; --no-classify needs neither).
"""

import argparse
import json
import sys

import cv2
from scenedetect import detect, AdaptiveDetector, ContentDetector

from analyze_deadspace import get_duration, analyze
# Reuse the whole classify/merge machinery -- this module only swaps the sampler.
from segment_content import (
    Segment, OTHER, CATEGORIES, CLASSIFY_MODEL,
    _read_at, _encode_frame, classify_window,
    confirm_targets, smooth_labels, _coalesce, consolidate_segments, other_note,
)


# ------------------------------- shot detection -------------------------------
def detect_shots(path, window=None, detector="adaptive", min_shot=0.0) -> list:
    """Detect shots and return [(start, end)] clamped to `window`.

    AdaptiveDetector is the default: on this analog/VHS source it is robust to
    the grain that makes a fixed-threshold ContentDetector over-fire. Runs on the
    whole file (a 33-min tape is ~20s because PySceneDetect downscales) and then
    clamps to the content window.

    `min_shot` coalesces micro-shots (rapid-cut montages -- e.g. a 15-cut parade
    in 20s) into their predecessor BEFORE classifying: those cuts are almost
    never separate content types, so absorbing them cuts API calls and label
    noise at a sub-`min_shot` cost in boundary precision.
    """
    det = AdaptiveDetector() if detector == "adaptive" else ContentDetector(threshold=30.0)
    scenes = detect(path, det, show_progress=False)

    lo, hi = window if window else (0.0, get_duration(path))
    clamped = []
    for s, e in scenes:
        ss, ee = max(s.seconds, lo), min(e.seconds, hi)
        if ee - ss > 0.05:                       # keep shots that overlap the window
            clamped.append((ss, ee))
    if not clamped:                              # no cuts detected -> whole window is one shot
        return [(lo, hi)]

    if min_shot > 0:
        merged = []
        for ss, ee in clamped:
            if merged and (ee - ss) < min_shot:  # absorb this micro-shot into the previous
                merged[-1] = (merged[-1][0], ee)
            else:
                merged.append((ss, ee))
        clamped = merged
    return clamped


def sample_shot(cap, start, end, frames_per_shot=3, max_width=768) -> list:
    """Grab up to `frames_per_shot` JPEGs spread across one shot.

    A shot is a single continuous take, so a couple of frames capture it; short
    shots get fewer (~1 frame/sec, always >= 1). Returns [] if none decode.
    """
    dur = end - start
    n = max(1, min(frames_per_shot, round(dur / 2) or 1))
    jpegs = []
    for k in range(n):
        t = start + (k + 0.5) / n * dur
        bgr = _read_at(cap, t)
        if bgr is not None:
            j = _encode_frame(bgr, max_width)
            if j:
                jpegs.append(j)
    return jpegs


def label_shots(path, shots, model=CLASSIFY_MODEL, frames_per_shot=3,
                max_width=768, categories=None) -> list:
    """Classify each shot into the closed set (or OTHER). Returns
    [(shot_start, category, description)]. One VLM call per shot."""
    import anthropic
    client = anthropic.Anthropic()
    cap = cv2.VideoCapture(path)
    labeled = []
    for ss, ee in shots:
        jpegs = sample_shot(cap, ss, ee, frames_per_shot, max_width)
        if not jpegs:
            continue
        cat, desc = classify_window(client, jpegs, model, ee - ss, categories)
        labeled.append((ss, cat, desc))
        shown = cat + (f" ({desc})" if cat == OTHER and desc else "")
        print(f"  [{ss:8.2f} -> {ee:8.2f}] ({len(jpegs)}f) {shown}", file=sys.stderr)
    cap.release()
    return labeled


# -------------------------------- build timeline ------------------------------
def _shots_to_segments(shots, cats) -> list:
    """One Segment per shot (its own [start,end] + label), then coalesce adjacent
    equal labels. Unlike segment_content.merge_segments this keeps each shot's
    real end instead of the next sample time, so boundaries stay on true cuts."""
    segs = [Segment(ss, ee, lab) for (ss, ee), (_, lab) in zip(shots, cats)]
    return _coalesce(segs)


def segment_video_shots(path, *, detector="adaptive", min_shot=1.5,
                        frames_per_shot=3, max_width=768, model=CLASSIFY_MODEL,
                        classify=True, smooth=True, trim=True, max_segments=0,
                        min_seg=0.0, categories=None, min_windows=1):
    """Trim -> detect shots -> classify -> confirm -> smooth -> merge -> consolidate.

    Returns (duration, window, [Segment]). No boundary refinement: shot cuts are
    already the real boundaries. `min_windows` defaults to 1 (a single confident
    shot is trustworthy, unlike a single 30s grid window); raise it to require a
    target type to span consecutive shots. With classify=False, each shot is
    emitted as its own 'unclassified' segment (no label-merging) so the shot
    timeline is visible without an API key."""
    categories = categories or CATEGORIES
    duration = get_duration(path)
    window = (0.0, duration)
    if trim:
        prop = analyze(path)
        if prop.status == "OK" and prop.kept > 0:
            window = (prop.content_start, prop.content_end)

    shots = detect_shots(path, window, detector, min_shot)

    if not classify:
        # Show the raw shot timeline; don't merge (every shot is its own segment).
        segments = [Segment(ss, ee, "unclassified") for ss, ee in shots]
        return duration, window, segments

    labeled = label_shots(path, shots, model, frames_per_shot, max_width, categories)
    # Realign shots to the ones that actually classified (label_shots drops
    # unreadable shots), so _shots_to_segments zips matching pairs.
    kept = {round(ss, 2) for ss, _, _ in labeled}
    shots = [(ss, ee) for ss, ee in shots if round(ss, 2) in kept]

    cats = [(ss, c) for ss, c, _ in labeled]
    cats = confirm_targets(cats, categories, min_windows)
    if smooth:
        cats = smooth_labels(cats)
    segments = _shots_to_segments(shots, cats)
    segments = consolidate_segments(segments, max_segments, min_seg)
    for s in segments:                       # name 'other' spans with the model's guess
        if s.label == OTHER:
            s.note = other_note(s, labeled)
    return duration, window, segments


# ------------------------------------ main ------------------------------------
def main():
    p = argparse.ArgumentParser(description="Shot-first content-type segmentation.")
    p.add_argument("input_video")
    p.add_argument("-o", "--output", default="segments_shots.json")
    p.add_argument("--detector", choices=["adaptive", "content"], default="adaptive")
    p.add_argument("--min-shot", type=float, default=1.5,
                   help="Coalesce shots shorter than this into the previous one "
                        "before classifying (0 = classify every micro-shot).")
    p.add_argument("--frames-per-shot", type=int, default=3)
    p.add_argument("--max-width", type=int, default=768)
    p.add_argument("--model", default=CLASSIFY_MODEL)
    p.add_argument("--no-classify", action="store_true",
                   help="Detect shots only; skip the Claude call (no API needed).")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--no-trim", action="store_true")
    p.add_argument("--categories",
                   help="Comma-separated closed set (default: %s)." % ", ".join(CATEGORIES))
    p.add_argument("--max-segments", type=int, default=0)
    p.add_argument("--min-seg", type=float, default=0.0,
                   help="Absorb segments shorter than this into a neighbor (0 = off).")
    p.add_argument("--min-windows", type=int, default=1,
                   help="A target type must hold this many consecutive shots (1 = "
                        "accept single-shot hits).")
    args = p.parse_args()

    categories = ([c.strip() for c in args.categories.split(",")]
                  if args.categories else CATEGORIES)

    duration, window, segments = segment_video_shots(
        args.input_video, detector=args.detector, min_shot=args.min_shot,
        frames_per_shot=args.frames_per_shot, max_width=args.max_width,
        model=args.model, classify=not args.no_classify, smooth=not args.no_smooth,
        trim=not args.no_trim, max_segments=args.max_segments, min_seg=args.min_seg,
        categories=categories, min_windows=args.min_windows,
    )

    print(f"\nInput: {args.input_video}  (duration {duration:.2f}s)")
    print(f"Content window (after trim): {window[0]:.2f}s -> {window[1]:.2f}s")
    print(f"Segments ({len(segments)}):")
    for s in segments:
        label = f"{s.label} ({s.note})" if s.note else s.label
        print(f"  [{s.start:8.2f} -> {s.end:8.2f}]  ({s.duration:6.1f}s)  {label}")

    out = {
        "video": args.input_video,
        "duration": round(duration, 2),
        "content_window": [round(window[0], 2), round(window[1], 2)],
        "detector": args.detector,
        "categories": categories,
        "segments": [{"start": round(s.start, 2), "end": round(s.end, 2),
                      "label": s.label, **({"note": s.note} if s.note else {})}
                     for s in segments],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(segments)} segments to {args.output}")


if __name__ == "__main__":
    main()