"""Color-bar (SMPTE/EBU test pattern) detector.

Distinguishes a color-bars frame from program content -- including *static*
program (a held neon-sign shot, a dark hallway) which `freezedetect` wrongly
calls "dead". Bars have a structure program lacks: a row of highly-saturated
VERTICAL bands, each near-constant top-to-bottom.

Three signals per frame (computed on the upper ~65%, to skip the SMPTE bottom
strip and any burned-in captions):
  - sat          : mean saturation (bars are vivid; dark/skin program is not)
  - vert_uniform : columns are near-constant top-to-bottom (bands), 0..1
  - bands        : number of distinct vertical color bands (bars ≈ 6-9;
                   a uniform blue VCR screen = 1; busy program = many/irregular)

A frame is bars if sat & vert_uniform are high AND bands is in a bars-like range.

CLI:
    python detect_bars.py frame.png                 # score one image
    python detect_bars.py video.mp4                 # scan first 90s, report bars
    python detect_bars.py video.mp4 --window 0      # scan whole file
"""

import argparse
import cv2
import numpy as np

# --- tunables (set against real frames below) ---
SAT_MIN = 0.55          # mean saturation floor (bars are very vivid)
VERT_UNIFORM_MIN = 0.70  # column-constancy floor (bars are very uniform top-to-bottom)
BANDS_MIN, BANDS_MAX = 5, 12
EDGE_THR = 42.0         # per-column color-change magnitude that marks a band edge


def bars_score(bgr) -> dict:
    h, w = bgr.shape[:2]
    roi = bgr[: max(1, int(h * 0.65)), :]
    roi = cv2.resize(roi, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0

    sat = float(S.mean())

    # vertical uniformity: low per-column std (top-to-bottom constant) -> ~1.0
    col_std = float(np.mean(V.std(axis=0) + S.std(axis=0)))
    vert_uniform = max(0.0, 1.0 - 2.5 * col_std)

    # banding: per-column mean BGR, count strong horizontal transitions
    col_color = roi.astype(np.float32).mean(axis=0)          # (160, 3)
    dif = np.abs(np.diff(col_color, axis=0)).sum(axis=1)     # (159,)
    edges = np.where(dif > EDGE_THR)[0]
    bands = 1
    if len(edges):
        bands = 1 + 1 + int(np.sum(np.diff(edges) > 2))     # merge adjacent edge px

    is_bars = (sat >= SAT_MIN and vert_uniform >= VERT_UNIFORM_MIN
               and BANDS_MIN <= bands <= BANDS_MAX)
    return {"sat": round(sat, 3), "vert_uniform": round(vert_uniform, 3),
            "bands": bands, "is_bars": is_bars}


def scan_video(path, window=90.0, step=1.0):
    """Sample frames every `step` s over the first `window` s (0 = whole file),
    score each, and return (bars_timestamps, all_results).

    Uses ffmpeg to extract sampled frames in a SINGLE streaming pass (fps filter),
    instead of per-sample cv2 seeking. Random seeking (cap.set POS_MSEC) is very
    slow on large H.264 files — a full-file scan of a 1GB video that way can take
    many minutes. Streaming decode at a low sample rate is bounded and fast."""
    import subprocess
    # Determine duration/end window via ffprobe (no full decode)
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            stderr=subprocess.DEVNULL).decode().strip()
        dur = float(out) if out and out != "N/A" else 0.0
    except Exception:
        dur = 0.0
    end = dur if window == 0 else min(window, dur or window)

    # Extract one frame every `step` seconds (fps=1/step), scaled small, as raw
    # RGB24 over a pipe. Trim to the window with -t.
    w, h = 160, 90
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", path]
    if end > 0:
        cmd += ["-t", f"{end:.3f}"]
    cmd += ["-vf", f"fps=1/{step},scale={w}:{h}", "-pix_fmt", "bgr24",
            "-f", "rawvideo", "pipe:1"]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    bars_ts, results = [], []
    idx = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        t = round(idx * step, 1)
        # frame is already 160x90; bars_score resizes/crops internally but is safe
        sc = bars_score(frame)
        results.append((t, sc))
        if sc["is_bars"]:
            bars_ts.append(t)
        idx += 1
    proc.stdout.close()
    proc.wait()
    return bars_ts, results


def bars_intervals(path, window=150.0, step=1.0, min_len=2.0):
    """Color-bar time intervals in the first `window` seconds, as [(start,end)].
    Drops single-sample blips shorter than `min_len` s (e.g. a lone bar-like frame).

    Each detected sample represents a `step`-second window, so an interval's end
    is extended by one step; this also lets coarse-step scans (step >= min_len)
    still report a run from as few as one positive sample."""
    bars_ts, _ = scan_video(path, window=window, step=step)
    intervals = []
    for t in bars_ts:
        if intervals and t - intervals[-1][1] <= 2 * step:
            intervals[-1][1] = t
        else:
            intervals.append([t, t])
    # Extend each interval end by one step (the sample covers up to the next probe)
    out = []
    for a, b in intervals:
        b_ext = b + step
        if b_ext - a >= min_len:
            out.append((a, b_ext))
    return out


def main():
    p = argparse.ArgumentParser(description="Detect SMPTE/EBU color bars.")
    p.add_argument("path", help="image or video file")
    p.add_argument("--window", type=float, default=90.0, help="seconds to scan (0=whole file)")
    p.add_argument("--step", type=float, default=1.0, help="sample interval (s)")
    p.add_argument("--verbose", action="store_true", help="print every sampled frame")
    args = p.parse_args()

    img = cv2.imread(args.path)
    if img is not None:  # single image
        print(f"{args.path}: {bars_score(img)}")
        return

    bars_ts, results = scan_video(args.path, args.window, args.step)
    if args.verbose:
        for t, sc in results:
            print(f"  t={t:6.1f}  {sc}")
    # collapse bars timestamps into intervals (gap <= 2*step)
    intervals = []
    for t in bars_ts:
        if intervals and t - intervals[-1][1] <= 2 * args.step:
            intervals[-1][1] = t
        else:
            intervals.append([t, t])
    frac = len(bars_ts) / max(1, len(results))
    print(f"{args.path}: bars in {len(bars_ts)}/{len(results)} sampled frames "
          f"({frac:.0%}); intervals={[(a, b) for a, b in intervals]}")


if __name__ == "__main__":
    main()
