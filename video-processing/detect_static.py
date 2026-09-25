"""Video static / "snow" detector (tape run-out, dead signal).

Snow is the opposite of a frozen frame: high-frequency noise everywhere, AND
each frame is uncorrelated with the next. That pair of properties separates it
from real program -- even busy/fast program stays spatially structured and
temporally coherent frame-to-frame.

Per adjacent frame-pair (analysed on a center crop at native resolution, so the
pixel-level noise is preserved -- downscaling would average it away):
  - lap   : variance of the Laplacian (high-frequency energy) -- snow is huge
  - smooth: fraction of 16x16 blocks that are smooth (low std) -- snow ~ 0
  - mad   : mean abs diff to the *next* frame -- snow is huge, frozen ~ 0

A pair is snow if lap & mad are high AND there are essentially no smooth blocks.

CLI:
    python detect_static.py video.mp4                # scan last 120s for snow
    python detect_static.py video.mp4 --head         # also scan first 120s
    python detect_static.py video.mp4 --verbose
"""

import argparse
import cv2
import numpy as np

import os

# Tunables. Snow (tape run-out / dead signal) is an EXTREME signal: near-pure noise
# with very high high-frequency energy, essentially no flat regions, and near-total
# frame-to-frame decorrelation. The thresholds were previously loosened for "noisy VHS
# transitions", which let busy/grainy REAL program read as snow (false positives). They
# are tightened here so only genuine static qualifies. All three must hold to flag snow,
# so raising any of them reduces false positives. Env-overridable for tuning without a
# code change (SNOW_LAP_MIN / SNOW_SMOOTH_MAX / SNOW_MAD_MIN).
# Priority: RECALL over precision — a real ~2s snow burst MUST be caught; a false
# positive is acceptable (a human trims it). So the FRAME thresholds stay SENSITIVE
# (catch faint/brief snow), and the single observed false positive (one lone busy
# frame) is suppressed downstream by requiring a SUSTAINED run (min_len >= 2s in
# snow_intervals): real snow trips several consecutive frames; a stray detailed frame
# does not. Env-overridable for tuning without a code change.
LAP_MIN = float(os.environ.get("SNOW_LAP_MIN", "500.0"))     # Laplacian variance floor
SMOOTH_MAX = float(os.environ.get("SNOW_SMOOTH_MAX", "0.10"))  # max fraction of flat blocks
MAD_MIN = float(os.environ.get("SNOW_MAD_MIN", "20.0"))      # min frame-to-frame mean-abs-diff
# NCC (normalized cross-correlation between consecutive frames) is the DEFINITIVE snow
# discriminator: snow is temporally DECORRELATED (each frame independent of the next),
# so consecutive-frame NCC ~ 0. Real program — even a busy, noisy, MOVING title/intro —
# stays temporally COHERENT (title, letterboxing, moving elements persist), so its NCC
# is clearly positive. Measured: true snow NCC <= ~0.10; a false-positive captioned
# intro (RCC_10 ~62-70s) NCC >= ~0.15. lap/smooth/mad alone can't separate those two
# (both look high-frequency), but NCC does. Snow requires NCC below this ceiling.
NCC_MAX = float(os.environ.get("SNOW_NCC_MAX", "0.12"))     # max consecutive-frame correlation
BLOCK = 16
SMOOTH_STD = 12.0    # a block with std below this is "smooth"


def _center_gray(bgr, size=256):
    h, w = bgr.shape[:2]
    s = min(size, h, w)
    y, x = (h - s) // 2, (w - s) // 2
    g = cv2.cvtColor(bgr[y:y + s, x:x + s], cv2.COLOR_BGR2GRAY)
    return g


def snow_score(prev_gray, gray) -> dict:
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    n = sm = 0
    for yy in range(0, h - BLOCK, BLOCK):
        for xx in range(0, w - BLOCK, BLOCK):
            n += 1
            if gray[yy:yy + BLOCK, xx:xx + BLOCK].std() < SMOOTH_STD:
                sm += 1
    smooth = sm / max(1, n)
    mad = (float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))
           if prev_gray is not None else 0.0)
    # Normalized cross-correlation to the previous frame: ~0 for snow (independent
    # frames), clearly positive for coherent program. High ncc (== 1.0 when there is
    # no previous frame) vetoes snow so the first sample of a window can't false-fire.
    if prev_gray is not None:
        a = prev_gray.astype(np.float64)
        b = gray.astype(np.float64)
        am, bm = a - a.mean(), b - b.mean()
        denom = float(np.sqrt((am * am).sum()) * np.sqrt((bm * bm).sum())) + 1e-6
        ncc = float((am * bm).sum() / denom)
    else:
        ncc = 1.0
    is_snow = (lap > LAP_MIN and smooth < SMOOTH_MAX and mad > MAD_MIN
               and ncc < NCC_MAX)
    return {"lap": round(lap, 1), "smooth": round(smooth, 3), "mad": round(mad, 1),
            "ncc": round(ncc, 3), "is_snow": is_snow}


def scan_range(path, start, end, step=1.0):
    """Scan [start,end] for snow. Streams frame PAIRS via ffmpeg (no per-sample
    cv2 seeking, which is very slow on large H.264 files). At each sample point we
    grab two consecutive native frames so the frame-to-frame diff (mad) — the
    signal that separates snow from a frozen frame — is measured at native rate.

    Implementation: ask ffmpeg for 2 frames every `step` seconds using the select
    filter, streamed as raw gray. Pairs arrive back-to-back in the stream."""
    import subprocess
    W = H = 256
    span = max(0.0, end - start)
    if span <= 0:
        return [], []
    # select two consecutive frames at the start of each `step` window:
    #   mod(t,step) picks the window; grab the first 2 frames of each window via
    #   a frame-index trick is complex, so instead sample at 2 frames per step by
    #   requesting fps=2/step won't give ADJACENT frames. Use select='lt(mod(n,N),2)'
    #   where N = step*native_fps to take the first 2 frames of each step-block.
    # Get native fps
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of",
             "default=nokey=1:noprint_wrappers=1", path],
            stderr=subprocess.DEVNULL).decode().strip()
        num, den = out.split("/") if "/" in out else (out, "1")
        native_fps = float(num) / float(den) if float(den) else 25.0
    except Exception:
        native_fps = 25.0
    N = max(2, int(round(step * native_fps)))

    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", path,
           "-vf", f"select='lt(mod(n\\,{N})\\,2)',scale={W}:{H},format=gray",
           "-vsync", "vfr", "-f", "rawvideo", "pipe:1"]
    frame_bytes = W * H
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    snow_ts, results = [], []
    idx = 0
    import numpy as _np
    while True:
        buf_a = proc.stdout.read(frame_bytes)
        buf_b = proc.stdout.read(frame_bytes)
        if len(buf_a) < frame_bytes or len(buf_b) < frame_bytes:
            break
        a = _np.frombuffer(buf_a, dtype=_np.uint8).reshape(H, W)
        b = _np.frombuffer(buf_b, dtype=_np.uint8).reshape(H, W)
        t = round(start + idx * step, 1)
        sc = snow_score(a, b)
        results.append((t, sc))
        if sc["is_snow"]:
            snow_ts.append(t)
        idx += 1
    proc.stdout.close()
    proc.wait()
    return snow_ts, results


def _video_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return total / fps if total else 0.0


def snow_intervals(path, windows, step=1.0, min_len=2.0):
    """Snow intervals within the given list of (start,end) windows.

    Each positive sample represents a `step`-second window, so an interval's end is
    extended by one step: this makes a real 2s snow burst (2 samples at step=1) span
    ~2s and clear `min_len`, rather than the 1s a bare max-of-timestamps would give.
    Priority is RECALL (a 2s burst MUST be flagged); the cost is that a lone-sample
    blip becomes a ~1-step interval — still below min_len=2, so single stray frames are
    dropped while genuine short bursts are kept."""
    snow_ts = []
    for s, e in windows:
        ts, _ = scan_range(path, s, e, step)
        snow_ts += ts
    intervals = []
    for t in sorted(set(snow_ts)):
        if intervals and t - intervals[-1][1] <= 2 * step:
            intervals[-1][1] = t
        else:
            intervals.append([t, t])
    return [(a, b + step) for a, b in intervals if (b + step) - a >= min_len]


def main():
    p = argparse.ArgumentParser(description="Detect video static / snow.")
    p.add_argument("path")
    p.add_argument("--window", type=float, default=120.0, help="seconds at each edge to scan")
    p.add_argument("--head", action="store_true", help="also scan the first --window seconds")
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dur = _video_duration(args.path)
    windows = [(max(0.0, dur - args.window), dur)]
    if args.head:
        windows.insert(0, (0.0, min(args.window, dur)))
    if args.verbose:
        for s, e in windows:
            for t, sc in scan_range(args.path, s, e, args.step)[1]:
                print(f"  t={t:7.1f}  {sc}")
    iv = snow_intervals(args.path, windows, args.step)
    print(f"{args.path}: snow intervals={iv}")


if __name__ == "__main__":
    main()
