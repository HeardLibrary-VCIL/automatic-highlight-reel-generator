"""Detect where a fixed set of content types occur in a video.

This is stage 2 of the pipeline: run it on the DEAD-SPACE-TRIMMED output of
analyze_deadspace.py. It splits the program into N-second windows, samples
several frames across each (one still is too ambiguous -- motion and variety are
what tell types apart), classifies each window into EXACTLY ONE of a CLOSED set
of target types (CATEGORIES: dance performance, football game, television
network, interview, political ad) or 'other', then merges consecutive same-type
windows into contiguous [start, end, type] segments.

A closed set is the point: it stops the model inventing near-synonym labels, so
merging equal labels already yields a small timeline -- no artificial segment
cap needed. Types that are visually distinctive (dance, football) classify well;
speech-defined ones (interview vs political ad) are harder from vision alone.

Design mirrors analyze_deadspace.py on purpose:
  - sample -> classify -> merge, the same detect -> propose shape.
  - merge_segments() plays the role merge_runs() plays there: it collapses a
    per-window signal into contiguous runs.

Why a model (and not ffmpeg signals): telling "interview" apart from "political
ad" is content understanding, which the model-free detectors can't do. We use
Claude's multimodal API via Amazon Bedrock so there
is no GPU to run -- the model runs on Bedrock's side, this process only needs
a CPU + network. Open-ended labels: the model names the type; a running
vocabulary of already-used labels is fed back each call to curb label drift.

Output: a segments.json timeline. No video is cut here.

Setup (one-time): model calls go through Amazon Bedrock (see bedrock.py), so no
ANTHROPIC_API_KEY -- reuse the project AWS creds:
    pip install anthropic            # not in requirements-trim.txt yet
    export AWS_SHARED_CREDENTIALS_FILE=../.env AWS_PROFILE=337513903342_PowerUserAccess

Runs the head/tail dead-space trim first (analyze_deadspace) and samples only
the resulting content window, then caps the timeline at --max-segments (default
6) by folding short inserts into their longer neighbors -- open-ended labels
otherwise fragment into many near-synonym spans. Coarse sampling snaps each
boundary to the grid, so a final pass bisects every boundary (forced A-or-B
frame checks) down to --precision seconds instead of the sampling interval.

Usage:
    python segment_content.py video.mp4                      # trim + segment -> segments.json
    python segment_content.py video.mp4 --max-segments 6 --interval 20
    python segment_content.py video.mp4 --no-trim            # segment the whole file
    python segment_content.py video.mp4 --no-classify        # sampling only (no API)

Future work (kept out of this minimal version): batch several frames per API
call to cut cost/latency; feed the subtitle/audio transcript of each window
alongside the frame (decisive for interview vs ad vs talk show).
"""

import argparse
import base64
import json
import sys
from collections import Counter
from dataclasses import dataclass

import cv2

# Reuse the trimmer directly: get_duration for probing, analyze for the
# head/tail dead-space window so we segment only the real program.
from analyze_deadspace import get_duration, analyze
# All model calls go through Amazon Bedrock (project AWS creds, no ANTHROPIC_API_KEY).
from bedrock import make_client, BEDROCK_MODEL

CLASSIFY_MODEL = BEDROCK_MODEL  # Bedrock Haiku 4.5 (vision); see bedrock.py for access notes


@dataclass
class Segment:
    start: float
    end: float
    label: str
    note: str = ""     # for 'other' spans: the model's free-text suggestion(s)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ------------------------------- sampling (cv2) -------------------------------
def _read_at(cap, t):
    """Decode the single frame nearest timestamp t (seconds). None on failure."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, bgr = cap.read()
    return bgr if ok else None


def _encode_frame(bgr, max_width) -> str:
    """Downscale to <= max_width wide and return a base64 JPEG ('' on failure)."""
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, round(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def sample_windows(path, interval=30.0, frames_per_window=3, max_width=768,
                   window=None) -> list:
    """Split the (trimmed) span into `interval`-second windows; grab several
    frames spread across each so the classifier sees motion, not one still.

    A single frame is ambiguous (a slow B-roll shot reads as a screensaver, an
    ad frame as an interview); a few frames across the window disambiguate it.
    Returns [(window_start, [jpeg_b64, ...])] with times in the ORIGINAL timeline.
    """
    start, end = window if window else (0.0, get_duration(path))
    cap = cv2.VideoCapture(path)
    windows, t = [], start
    while t < end:
        span_end = min(t + interval, end)
        jpegs = []
        for k in range(frames_per_window):
            ft = t + (k + 0.5) / frames_per_window * (span_end - t)
            bgr = _read_at(cap, ft)
            if bgr is not None:
                j = _encode_frame(bgr, max_width)
                if j:
                    jpegs.append(j)
        if jpegs:
            windows.append((round(t, 2), jpegs))
        elif t == start:
            break  # unreadable from the very first window -> bad file/window
        t += interval
    cap.release()
    return windows


# ----------------------------- classify (Claude) ------------------------------
# Closed taxonomy: each window is put into EXACTLY ONE of these, or 'other'.
# This is the whole point of the tool -- detect where these specific types occur,
# not describe the video. Edit the list (or pass --categories) to change targets.
CATEGORIES = ["dance performance", "football game", "tv show", "interview",
              "political ad", "public service announcement", "music performance"]
OTHER = "other"

# Optional clarifications shown after a category in the prompt. Broadens the
# fuzzy ones (dance covers concerts/parties) and separates the two ad-like types.
CATEGORY_HINTS = {
    "dance performance": "any real dancing on screen -- a stage or dance "
                         "performance, a party, or a crowd dancing",
    "political ad": "a campaign spot for a candidate, party, or ballot measure",
    "public service announcement": "a non-commercial awareness or advocacy "
                                   "message (health, safety, social causes), "
                                   "including animated or dramatized ones",
}

PROMPT = (
    "These {n} images are frames sampled in time order across one ~{interval:.0f}s "
    "window of a video. Look at them together -- motion between frames, setting, "
    "on-screen graphics -- and classify the window into EXACTLY ONE of these "
    "content types:\n{categories}\n"
    "If it matches none of them, use 'other'.\n"
    "Then, in 1-3 words, describe what the window actually shows.\n"
    "Reply on ONE line as: <category or other> / <short description>\n"
    "Example: interview / studio interview   -or-   other / weather forecast"
)


def _render_categories(categories) -> str:
    """Bullet each category, appending its CATEGORY_HINTS gloss when present."""
    return "\n".join(f"- {c}" + (f": {CATEGORY_HINTS[c]}" if c in CATEGORY_HINTS else "")
                     for c in categories)


def _canonical(text, categories) -> str:
    """Map the model's free-text reply onto one of `categories`, else OTHER."""
    r = text.strip().lower()
    for c in categories:              # exact / substring match
        if c in r or r in c:
            return c
    for c in categories:              # looser: any word of the category appears
        if any(w in r for w in c.split()):
            return c
    return OTHER


def classify_window(client, jpegs, model=CLASSIFY_MODEL, interval=30.0,
                    categories=None):
    """Classify a window's frames. Returns (category, description) where category
    is one of `categories` or OTHER, and description is the model's free-text
    suggestion of what it is (used to label 'other' spans)."""
    categories = categories or CATEGORIES
    prompt = PROMPT.format(n=len(jpegs), interval=interval,
                           categories=_render_categories(categories))
    content = [{"type": "image", "source": {"type": "base64",
                                            "media_type": "image/jpeg", "data": j}}
               for j in jpegs]
    content.append({"type": "text", "text": prompt})
    from bedrock import create_with_retry
    resp = create_with_retry(
        client, model=model, max_tokens=24,
        messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    left, _, right = text.partition("/")
    category = _canonical(left, categories)
    desc = right.strip().lower()[:40]
    if category == OTHER and not desc:   # model described without the "category /" prefix
        desc = left.strip().lower()[:40]
    return category, desc


def label_windows(windows, model=CLASSIFY_MODEL, interval=30.0, categories=None) -> list:
    """Classify each window. Returns [(t, category, description)].
    The Bedrock client is built here (not at import) so --no-classify never
    touches the anthropic SDK / AWS creds."""
    client = make_client()
    labeled = []
    for t, jpegs in windows:
        cat, desc = classify_window(client, jpegs, model, interval, categories)
        labeled.append((t, cat, desc))
        shown = f"{cat}" + (f" ({desc})" if cat == OTHER and desc else "")
        print(f"  [{t:8.2f}s] ({len(jpegs)}f) {shown}", file=sys.stderr)
    return labeled


# --------------------------- boundary refinement ------------------------------
# The coarse pass snaps each segment boundary to the sample grid, so a true
# content change between two samples is off by up to `interval`. We locate it
# precisely by BISECTING the (b - interval, b) window around each boundary,
# asking a forced A-or-B choice at each probe frame -- far cheaper than dense
# sampling (~log2(interval/precision) calls per boundary) and only run on the
# few final boundaries.
AB_PROMPT = (
    "This is one frame from a video. Which of these two content types does it show?\n"
    "A: {a}\nB: {b}\n"
    "Reply with only the single letter A or B."
)


def classify_ab(client, jpeg_b64, a_label, b_label, model=CLASSIFY_MODEL) -> str:
    """Forced choice: does this frame show a_label (A) or b_label (B)?
    Returns whichever label string the model picked."""
    from bedrock import create_with_retry
    resp = create_with_retry(
        client,
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": jpeg_b64}},
            {"type": "text", "text": AB_PROMPT.format(a=a_label, b=b_label)},
        ]}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    r = text.upper()
    if r.startswith("A"):
        return a_label
    if r.startswith("B"):
        return b_label
    low = text.lower()  # fallback: model echoed a label instead of a letter
    return b_label if (b_label in low and a_label not in low) else a_label


def refine_boundaries(path, segments, *, interval, model=CLASSIFY_MODEL,
                      precision=2.0, max_width=768) -> list:
    """Bisect each internal boundary to within `precision` seconds, in place.

    For adjacent segments A|B meeting at b, the transition lies in (b-interval, b].
    Probe the midpoint, ask classify_ab: if it still looks like A, the change is
    later (raise the low bound); if B, earlier (lower the high bound). Converge,
    then set the shared boundary to the first B-looking time."""
    client = make_client()
    cap = cv2.VideoCapture(path)
    for i in range(len(segments) - 1):
        seg, nxt = segments[i], segments[i + 1]
        lo, hi = max(seg.start, seg.end - interval), seg.end
        while hi - lo > precision:
            m = (lo + hi) / 2.0
            bgr = _read_at(cap, m)
            if bgr is None:
                break
            lab = classify_ab(client, _encode_frame(bgr, max_width), seg.label, nxt.label, model)
            if lab == seg.label:
                lo = m
            else:
                hi = m
        new_b = round(hi, 2)
        print(f"  boundary {seg.label} -> {nxt.label}: {seg.end:.1f}s => {new_b:.1f}s",
              file=sys.stderr)
        seg.end = nxt.start = new_b
    cap.release()
    return segments


# ------------------------------- merge segments -------------------------------
def confirm_targets(labeled, categories, min_windows=2) -> list:
    """Demote any TARGET-category window that is not part of a run of at least
    `min_windows` consecutive identical target labels down to OTHER.

    A single flickered target window (e.g. one frame read as 'political ad'
    between an interview and a report) is a false positive; smoothing only
    catches it when both neighbors match, so this corroboration rule handles the
    interview|target|other case smoothing misses. OTHER runs are never demoted."""
    targets = set(categories)
    cats = [c for _, c in labeled]
    keep, i, n = [True] * len(cats), 0, len(cats)
    while i < n:
        j = i
        while j < n and cats[j] == cats[i]:
            j += 1
        if cats[i] in targets and (j - i) < min_windows:
            for k in range(i, j):
                keep[k] = False
        i = j
    return [(t, c if keep[idx] else OTHER) for idx, (t, c) in enumerate(labeled)]


def other_note(seg, per_window) -> str:
    """The model's suggested description(s) for an 'other' span: the 1-2 most
    common per-window descriptions inside it."""
    descs = [d for (t, c, d) in per_window if seg.start <= t < seg.end and d]
    return ", ".join(d for d, _ in Counter(descs).most_common(2)) if descs else ""


def smooth_labels(labeled) -> list:
    """Replace an isolated single-window label that differs from both neighbors
    with the previous label (kills 1-frame classifier blips before merging)."""
    if len(labeled) < 3:
        return labeled
    out = [labeled[0]]
    for i in range(1, len(labeled) - 1):
        t, lab = labeled[i]
        prev, nxt = labeled[i - 1][1], labeled[i + 1][1]
        out.append((t, prev if lab != prev and prev == nxt else lab))
    out.append(labeled[-1])
    return out


def merge_segments(labeled, end_time) -> list:
    """Collapse consecutive same-label windows into [start, end, label] segments.

    Window i spans [t_i, t_{i+1}); the last spans to `end_time` (the content-end,
    not the file duration, when trimming). Same as the run-merging in
    analyze_deadspace.merge_runs, specialized to equal labels."""
    if not labeled:
        return []
    segments = []
    for i, (t, lab) in enumerate(labeled):
        end = labeled[i + 1][0] if i + 1 < len(labeled) else end_time
        if segments and segments[-1].label == lab:
            segments[-1].end = end
        else:
            segments.append(Segment(t, end, lab))
    return segments


def _coalesce(segs) -> list:
    """Merge adjacent segments that share a label into one (fresh Segment objects)."""
    out = []
    for s in segs:
        if out and out[-1].label == s.label:
            out[-1].end = s.end
        else:
            out.append(Segment(s.start, s.end, s.label))
    return out


def consolidate_segments(segments, max_segments=0, min_seg=0.0) -> list:
    """Clean up a timeline: absorb spans shorter than `min_seg` into their LONGER
    neighbor, then (only if max_segments > 0) merge the shortest spans until the
    count is within max_segments.

    With a closed taxonomy both are usually off (0): merging equal labels already
    yields few segments, and a hard cap would fabricate mega-blocks. Turn min_seg
    up to drop brief blips, or max_segments up to force a coarser view.

    Deterministic and free (no API). An absorbed span takes its bigger neighbor's
    label."""
    segs = _coalesce(list(segments))

    def absorb(i):
        left = segs[i - 1] if i > 0 else None
        right = segs[i + 1] if i < len(segs) - 1 else None
        into_left = right is None or (left is not None and left.duration >= right.duration)
        if into_left:
            left.end = segs[i].end
        else:
            right.start = segs[i].start
        segs.pop(i)

    # 1) absorb every span shorter than min_seg, shortest first
    while min_seg and len(segs) > 1:
        i = min(range(len(segs)), key=lambda j: segs[j].duration)
        if segs[i].duration >= min_seg:
            break
        absorb(i)
        segs[:] = _coalesce(segs)
    # 2) optional hard cap: merge the shortest span until within max_segments
    while max_segments and len(segs) > max_segments:
        i = min(range(len(segs)), key=lambda j: segs[j].duration)
        absorb(i)
        segs[:] = _coalesce(segs)
    return segs


# ------------------------------- callable API ---------------------------------
def segment_video(path, *, interval=30.0, frames_per_window=3, max_width=768,
                  model=CLASSIFY_MODEL, classify=True, smooth=True, trim=True,
                  max_segments=0, min_seg=0.0, refine=True, precision=2.0,
                  categories=None, min_windows=2):
    """Trim -> sample windows -> classify (closed set) -> confirm -> merge -> refine.

    Returns (duration, window, [Segment]) where window=(content_start, content_end).
    With trim=True, the head/tail dead-space window from analyze_deadspace bounds
    the sampling. Each window is classified into one of `categories` or 'other';
    a target category must hold for >= min_windows consecutive windows to survive
    (see confirm_targets). 'other' spans carry the model's free-text suggestion as
    Segment.note. With classify=False, every window is labeled 'unclassified'.
    max_segments/min_seg optionally coarsen; refine bisects each boundary to `precision`."""
    categories = categories or CATEGORIES
    duration = get_duration(path)
    window = (0.0, duration)
    if trim:
        prop = analyze(path)
        if prop.status == "OK" and prop.kept > 0:
            window = (prop.content_start, prop.content_end)
    windows = sample_windows(path, interval, frames_per_window, max_width, window)
    if classify:
        per_window = label_windows(windows, model, interval, categories)   # [(t, cat, desc)]
    else:
        per_window = [(t, "unclassified", "") for t, _ in windows]

    cats = [(t, c) for t, c, _ in per_window]
    # Confirm BEFORE smoothing: smoothing fills a lone 'other' between two equal
    # targets, which would fabricate a 2-window run out of two 1-window false
    # positives. Demoting lone targets first prevents that; smoothing then only
    # restores genuine one-window dropouts inside confirmed target runs.
    if classify:
        cats = confirm_targets(cats, categories, min_windows)
    if smooth:
        cats = smooth_labels(cats)
    segments = merge_segments(cats, window[1])
    segments = consolidate_segments(segments, max_segments, min_seg)
    if classify and refine and len(segments) > 1:
        segments = refine_boundaries(path, segments, interval=interval, model=model,
                                     precision=precision, max_width=max_width)
    # Label each 'other' span with what the model suggested it is.
    for s in segments:
        if s.label == OTHER:
            s.note = other_note(s, per_window)
    return duration, window, segments


# ------------------------------------ main ------------------------------------
def main():
    p = argparse.ArgumentParser(description="Segment a video into content-type spans.")
    p.add_argument("input_video")
    p.add_argument("-o", "--output", default="segments.json")
    p.add_argument("--interval", type=float, default=30.0,
                   help="Seconds per classification window.")
    p.add_argument("--frames-per-window", type=int, default=3,
                   help="Frames sampled across each window (motion beats one still).")
    p.add_argument("--max-width", type=int, default=768,
                   help="Downscale frames to this width before sending (token control).")
    p.add_argument("--model", default=CLASSIFY_MODEL)
    p.add_argument("--no-classify", action="store_true",
                   help="Sample and merge only; skip the Claude call (no API needed).")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--no-trim", action="store_true",
                   help="Segment the whole file; skip the head/tail dead-space trim.")
    p.add_argument("--categories",
                   help="Comma-separated closed set to detect (default: %s)."
                        % ", ".join(CATEGORIES))
    p.add_argument("--max-segments", type=int, default=0,
                   help="Optional cap on segment count (0 = no cap).")
    p.add_argument("--min-seg", type=float, default=0.0,
                   help="Absorb segments shorter than this into a neighbor (0 = off).")
    p.add_argument("--min-windows", type=int, default=2,
                   help="A target type must hold this many consecutive windows to "
                        "count (1 = accept single-window hits).")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip binary-search refinement of segment boundaries.")
    p.add_argument("--precision", type=float, default=2.0,
                   help="Refine each boundary to within this many seconds.")
    args = p.parse_args()

    categories = ([c.strip() for c in args.categories.split(",")]
                  if args.categories else CATEGORIES)

    duration, window, segments = segment_video(
        args.input_video, interval=args.interval,
        frames_per_window=args.frames_per_window, max_width=args.max_width,
        model=args.model, classify=not args.no_classify, smooth=not args.no_smooth,
        trim=not args.no_trim, max_segments=args.max_segments, min_seg=args.min_seg,
        refine=not args.no_refine, precision=args.precision, categories=categories,
        min_windows=args.min_windows,
    )

    print(f"\nInput: {args.input_video}  (duration {duration:.2f}s)")
    print(f"Content window (after trim): {window[0]:.2f}s -> {window[1]:.2f}s")
    for s in segments:
        label = f"{s.label} ({s.note})" if s.note else s.label
        print(f"  [{s.start:8.2f} -> {s.end:8.2f}]  ({s.duration:6.1f}s)  {label}")

    # Presence + total duration per target category (the "does it contain X" view).
    totals = {c: 0.0 for c in categories}
    for s in segments:
        if s.label in totals:
            totals[s.label] += s.duration
    print("\nDetected content types:")
    for c in categories:
        secs = totals[c]
        mark = "yes" if secs > 0 else " no"
        print(f"  [{mark}] {c:<22} {secs:7.1f}s")

    out = {
        "video": args.input_video,
        "duration": round(duration, 2),
        "content_window": [round(window[0], 2), round(window[1], 2)],
        "categories": categories,
        "present": {c: round(totals[c], 2) for c in categories if totals[c] > 0},
        "segments": [{"start": round(s.start, 2), "end": round(s.end, 2),
                      "label": s.label, **({"note": s.note} if s.note else {})}
                     for s in segments],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(segments)} segments to {args.output}")


if __name__ == "__main__":
    main()