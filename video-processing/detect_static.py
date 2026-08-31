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
LAP_MIN = 400.0      # Laplacian variance floor (lowered for noisy VHS transitions)
SMOOTH_MAX = 0.10    # max fraction of smooth (flat) blocks (raised for partial-signal fuzz)
MAD_MIN = 20.0       # min mean-abs-diff to the next frame (lowered for low-contrast noise)
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
