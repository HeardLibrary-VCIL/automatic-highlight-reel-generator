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

# tunables (set against synthetic snow vs real program below)
LAP_MIN = 600.0      # Laplacian variance floor
SMOOTH_MAX = 0.06    # max fraction of smooth (flat) blocks
MAD_MIN = 28.0       # min mean-abs-diff to the next frame
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
    is_snow = lap > LAP_MIN and smooth < SMOOTH_MAX and mad > MAD_MIN
    return {"lap": round(lap, 1), "smooth": round(smooth, 3), "mad": round(mad, 1), "is_snow": is_snow}


def scan_range(path, start, end, step=1.0):
    cap = cv2.VideoCapture(path)
    snow_ts, results = [], []
    t = max(0.0, start)
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok_a, a = cap.read()
        ok_b, b = cap.read()        # the very next frame (native-rate neighbor)
        if not (ok_a and ok_b):
            break
        sc = snow_score(_center_gray(a), _center_gray(b))
        results.append((round(t, 1), sc))
        if sc["is_snow"]:
            snow_ts.append(round(t, 1))
        t += step
    cap.release()
    return snow_ts, results


def _video_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return total / fps if total else 0.0


def snow_intervals(path, windows, step=1.0, min_len=2.0):
    """Snow intervals within the given list of (start,end) windows."""
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
    return [(a, b) for a, b in intervals if b - a >= min_len]


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
