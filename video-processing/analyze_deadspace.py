"""Probe a video for 'dead space' at the head and tail, using ffmpeg detectors.

Measurement + conservative trim proposal. It runs ffmpeg's black / freeze /
silence detectors, then proposes a single "content" interval
[content_start, content_end] by stripping the dead block at the very start and
end -- but anchored on BLACK and guarded against over-trimming.

Why black-anchored: across a diverse collection, generic "frozen" frames are a
dangerous signal -- static-camera footage, held graphics and community-bulletin
slates all read as frozen, so a freeze-driven trim eats real program. Black,
by contrast, is rarely sustained inside real program, so it is a safe place to
cut. freeze/silence are still used to establish that an edge region IS dead;
they just don't decide the cut point.

Detectors:
  - black   : near-black frames            (ffmpeg blackdetect)
  - freeze  : static / unchanging frames   (ffmpeg freezedetect)  -> bars, slates
  - silence : near-silent audio            (ffmpeg silencedetect)

Modes (--mode):
  - black  (default): cut at the black frame that ends the leading dead run /
                      begins the trailing dead run. Keeps held title cards.
  - static          : cut at the end/start of the whole static run (aggressive;
                      also removes held title/closing cards).

Guard rail (--min-keep): if the kept span would be a smaller fraction of the
file than this, the proposal is marked NEEDS_REVIEW and --apply refuses to run
(unless --force). This catches the catastrophic over-trims.

Usage:
    python analyze_deadspace.py video.mp4                 # report
    python analyze_deadspace.py video.mp4 --apply -o out.mp4
    python analyze_deadspace.py --from-log saved.probe.txt   # re-score offline
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Region:
    start: float
    end: float
    kind: str  # "black" | "freeze" | "silence"


@dataclass
class Run:
    start: float
    end: float
    blacks: list = field(default_factory=list)  # (start, end) of black sub-regions
    whites: list = field(default_factory=list)  # (start, end) of white sub-regions
    bars: list = field(default_factory=list)    # (start, end) of color-bar sub-regions
    snows: list = field(default_factory=list)   # (start, end) of video-static/snow sub-regions


@dataclass
class Proposal:
    """Result of analyzing a video: the proposed content window + verdict.

    This is the object callers (e.g. the ECS entry point main.py) consume instead
    of re-parsing CLI output."""
    duration: float
    content_start: float
    content_end: float
    status: str            # "OK" | "NEEDS_REVIEW"
    notes: list
    regions: list = field(default_factory=list)  # detected dead Regions (for logging)

    @property
    def kept(self) -> float:
        return self.content_end - self.content_start

    @property
    def kept_pct(self) -> float:
        return 100 * self.kept / self.duration if self.duration else 0.0

    @property
    def head_trim(self) -> float:
        return self.content_start

    @property
    def tail_trim(self) -> float:
        return self.duration - self.content_end


# ----------------------------- detection (ffmpeg) -----------------------------
def get_duration(path: str) -> float:
    # Some edited/split MP4s lack a container-level duration; fall back to the
    # video stream's duration, then to a decode pass that reports the end time.
    attempts = [
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", path],
    ]
    for cmd in attempts:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            if out and out != "N/A":
                return float(out)
        except Exception:
            pass
    # Last resort: decode once (with stats) and read ffmpeg's final reported time.
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", path, "-an", "-f", "null", "-"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if times:
        h, m, s = times[-1]
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"could not determine duration for {path}")


def _run_detector(path: str, args: list) -> str:
    # -nostdin + stdin=DEVNULL so ffmpeg never consumes the caller's stdin
    # (e.g. a `while read` loop feeding it a file list).
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path, *args, "-f", "null", "-"]
    return subprocess.run(cmd, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).stderr


def detect_black(path, min_dur, pic_th):
    log = _run_detector(path, ["-an", "-vf", f"blackdetect=d={min_dur}:pic_th={pic_th}"])
    return [Region(float(m.group("s")), float(m.group("e")), "black")
            for m in re.finditer(r"black_start:(?P<s>[\d.]+)\s+black_end:(?P<e>[\d.]+)", log)]


def detect_white(path, min_dur=0.5, white_th=0.98):
    """Detect sustained near-white frames using ffmpeg's negate + blackdetect trick.

    There's no built-in 'whitedetect' in ffmpeg, but negating the video turns
    white→black, so we can reuse blackdetect on the inverted signal. The pic_th
    parameter controls how white a frame must be (0.98 = 98% of pixels near-white
    after negate → near-black).
    """
    log = _run_detector(path, ["-an", "-vf",
                               f"negate,blackdetect=d={min_dur}:pic_th={white_th}"])
    return [Region(float(m.group("s")), float(m.group("e")), "white")
            for m in re.finditer(r"black_start:(?P<s>[\d.]+)\s+black_end:(?P<e>[\d.]+)", log)]


def detect_freeze(path, min_dur, noise_db, duration):
    log = _run_detector(path, ["-an", "-vf", f"freezedetect=n={noise_db}dB:d={min_dur}"])
    regions, start = [], None
    for m in re.finditer(r"freeze_(start|end):\s*(?P<t>[\d.]+)", log):
        if m.group(1) == "start":
            start = float(m.group("t"))
        elif start is not None:
            regions.append(Region(start, float(m.group("t")), "freeze"))
            start = None
    if start is not None:
        regions.append(Region(start, duration, "freeze"))
    return regions


def detect_silence(path, min_dur, noise_db, duration):
    log = _run_detector(path, ["-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}"])
    regions, start = [], None
    for m in re.finditer(r"silence_(start|end):\s*(?P<t>-?[\d.]+)", log):
        if m.group(1) == "start":
            start = max(0.0, float(m.group("t")))
        elif start is not None:
            regions.append(Region(start, float(m.group("t")), "silence"))
            start = None
    if start is not None:
        regions.append(Region(start, duration, "silence"))
    return regions


# ------------------------------- proposal logic -------------------------------
def merge_runs(regions, gap):
    """Merge dead regions into contiguous runs (bridging gaps <= `gap`).
    Each run remembers the black sub-regions inside it."""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r.start)

    def record(run, r):
        if r.kind == "black":
            run.blacks.append((r.start, r.end))
        elif r.kind == "white":
            run.whites.append((r.start, r.end))
        elif r.kind == "bars":
            run.bars.append((r.start, r.end))
        elif r.kind == "snow":
            run.snows.append((r.start, r.end))

    runs = [Run(ordered[0].start, ordered[0].end)]
    record(runs[0], ordered[0])
    for r in ordered[1:]:
        cur = runs[-1]
        if r.start - cur.end <= gap:
            cur.end = max(cur.end, r.end)
            record(cur, r)
        else:
            runs.append(Run(r.start, r.end))
            record(runs[-1], r)
    return runs


def propose_window(regions, duration, gap, edge_tol, min_keep, mode):
    """Returns (content_start, content_end, status, notes).

    Head and tail are treated ASYMMETRICALLY:
      - Head uses all signals -- freeze is needed to catch a bars/static leader
        that starts at t=0 (bars are neither black nor silent).
      - Tail EXCLUDES freeze -- closing program content (talking heads, held
        shots) is legitimately static, so letting freeze define the trailing
        run chains real content into "dead" and cuts the program short. Only
        black+silence may define the trailing dead run; the cut still anchors
        on black.
    """
    content_start, content_end = 0.0, duration
    notes = []

    all_runs = merge_runs(regions, gap)
    ns_runs = merge_runs([r for r in regions if r.kind != "freeze"], gap)  # black+silence

    head = next((r for r in all_runs if r.start <= edge_tol), None)
    tail = next((r for r in reversed(ns_runs) if r.end >= duration - edge_tol), None)

    # Whole-file-dead guard: the leading all-signal run reaches the end.
    if head is not None and head.end >= duration - edge_tol:
        notes.append(f"entire file reads as one continuous dead/static run "
                     f"({head.start:.1f}-{head.end:.1f}s); cannot locate program boundaries")
        return 0.0, duration, "NEEDS_REVIEW", notes

    if head:
        # Anchor the head cut on BLACK or on detected color BARS -- never on raw
        # freeze: a leading "frozen" run can be a static OPENING SHOT of real
        # program (e.g. RCC_161's held neon sign), and the bars detector is what
        # tells true bars apart from static program. Cut after whichever leader
        # element (black or bars) ends last.
        anchors = ([b[1] for b in head.blacks] + [b[1] for b in head.whites]
                   + [b[1] for b in head.bars]
                   + [b[1] for b in head.snows])  # end of last black/white/bars/snow in run
        if mode == "black" and anchors:
            content_start = max(anchors)
        elif mode == "black":
            notes.append(f"leading dead run 0->{head.end:.1f}s has no black/bars anchor; head not trimmed")
        else:  # static
            content_start = head.end
    if tail:
        # cut where the trailing dead block begins: the first black, white, OR snow.
        tail_anchors = ([b[0] for b in tail.blacks] + [b[0] for b in tail.whites]
                        + [b[0] for b in tail.snows])
        if mode == "black" and tail_anchors:
            content_end = min(tail_anchors)
        elif mode == "black":
            notes.append(f"trailing dead run {tail.start:.1f}s->end has no black/snow anchor; tail not trimmed")
        else:  # static
            content_end = tail.start

    kept = content_end - content_start
    status = "OK"
    if kept <= 0:
        status = "NEEDS_REVIEW"
        notes.append("kept span is empty")
    elif kept / duration < min_keep:
        status = "NEEDS_REVIEW"
        notes.append(f"kept only {100 * kept / duration:.0f}% of file (< {100 * min_keep:.0f}% floor)")
    return content_start, content_end, status, notes


# ------------------------------- offline re-score -----------------------------
def parse_log(path):
    """Reconstruct (duration, regions) from a saved probe .txt log."""
    text = open(path).read()
    m = re.search(r"\(duration ([\d.]+)s\)", text)
    duration = float(m.group(1)) if m else 0.0
    regions = [Region(float(s), float(e), k) for s, e, k in
               re.findall(r"\[\s*([\d.]+)\s*->\s*([\d.]+)\]\s+(black|white|freeze|silence|bars|snow)", text)]
    return duration, regions


# ------------------------------- callable API ---------------------------------
# These wrap the detect -> propose -> trim steps so other modules (e.g. the ECS
# entry point main.py) can call them directly instead of shelling out to the CLI.

def probe_video(input_video, *, min_dur=0.5, black_pic_th=0.995, black_min_dur=3.0,
                freeze_db=-30.0,
                silence_db=-30.0, bars_window=150.0, snow_window=150.0,
                no_black=False, no_white=False, no_freeze=False, no_silence=False,
                no_bars=False, no_snow=False):
    """Run the enabled detectors over a video. Returns (duration, [Region])."""
    duration = get_duration(input_video)
    regions = []
    if not no_black:
        regions += detect_black(input_video, black_min_dur, black_pic_th)
    if not no_white:
        regions += detect_white(input_video, black_min_dur)
    if not no_freeze:
        regions += detect_freeze(input_video, min_dur, freeze_db, duration)
    if not no_silence:
        regions += detect_silence(input_video, min_dur, silence_db, duration)
    if not no_bars:
        import detect_bars
        for s, e in detect_bars.bars_intervals(input_video, window=min(bars_window, duration)):
            regions.append(Region(s, e, "bars"))
    if not no_snow:
        import detect_static
        # For full-video analysis, scan the entire file for snow/static (not just edges).
        # Use step=2.0 for efficiency on long videos.
        snow_windows = [(0.0, duration)]
        for s, e in detect_static.snow_intervals(input_video, snow_windows, step=2.0):
            regions.append(Region(s, e, "snow"))
    return duration, regions


def analyze(input_video=None, *, from_log=None, mode="black", merge_gap=0.5,
            edge_tol=5.0, min_keep=0.5, **probe_kwargs) -> Proposal:
    """Probe a video (or re-score a saved log) and propose a content window.

    Does NOT write any video. Pass detector overrides through as keyword args
    (min_dur, freeze_db, no_bars, ...). Returns a Proposal."""
    if from_log:
        duration, regions = parse_log(from_log)
    else:
        if not input_video:
            raise ValueError("analyze() needs input_video or from_log")
        duration, regions = probe_video(input_video, **probe_kwargs)
    cs, ce, status, notes = propose_window(regions, duration, merge_gap, edge_tol, min_keep, mode)
    return Proposal(duration, cs, ce, status, notes, regions)


def apply_trim(input_video, output, content_start, content_end) -> str:
    """Cut [content_start, content_end] out of input_video and re-encode to output.
    Returns the output path."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{content_start}", "-to", f"{content_end}",
           "-i", input_video, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k", output]
    subprocess.run(cmd, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output


def trim_video(input_video, output, *, mode="black", force=False, **kwargs):
    """Convenience: analyze, then apply the trim iff status OK (or force).
    Returns (Proposal, wrote: bool)."""
    prop = analyze(input_video, mode=mode, **kwargs)
    wrote = False
    if prop.kept > 0 and (prop.status == "OK" or force):
        apply_trim(input_video, output, prop.content_start, prop.content_end)
        wrote = True
    return prop, wrote


# ------------------------------- full-video dead-space ------------------------
@dataclass
class DeadSpan:
    """A confirmed dead region anywhere in the video."""
    start: float
    end: float
    kind: str   # "black" | "bars" | "snow" | "mixed"


@dataclass
class ContentSpan:
    """A content region between dead spans."""
    start: float
    end: float


@dataclass
class FullVideoAnalysis:
    """Full-video dead-space analysis: dead spans + content spans covering 0→duration."""
    duration: float
    dead_spans: list       # [DeadSpan, ...]
    content_spans: list    # [ContentSpan, ...]
    status: str            # "OK" | "NEEDS_REVIEW"
    notes: list
    regions: list          # raw detector regions (for logging)


def find_all_dead_spans(regions, duration, *, merge_gap=2.0, min_dead_dur=3.0,
                        mode="black") -> list:
    """Find ALL dead spans throughout the video, not just head/tail.

    Uses the same black-anchored philosophy as the head/tail detector: a dead
    span must contain at least one black, bars, or snow region to qualify. Pure
    freeze or silence alone is not enough (could be a static shot or quiet moment
    in real program).

    Args:
        regions:     raw detector Regions from probe_video()
        duration:    total video duration
        merge_gap:   bridge dead regions separated by less than this (seconds)
        min_dead_dur: minimum duration for a dead span to be reported (seconds)
        mode:        "black" (require black/bars/snow anchor) or "static" (any signal)

    Returns:
        List of DeadSpan sorted by start time.
    """
    if not regions:
        return []

    # Merge ALL signals into runs (bridging small gaps)
    runs = merge_runs(regions, merge_gap)

    dead_spans = []
    for run in runs:
        run_dur = run.end - run.start
        if run_dur < min_dead_dur:
            continue

        if mode == "black":
            # Must have at least one black, white, bars, or snow anchor
            has_anchor = bool(run.blacks or run.whites or run.bars or run.snows)
            if not has_anchor:
                continue
            # Determine the kind based on what's dominant in this span
            if run.bars:
                kind = "bars"
            elif run.snows:
                # Snow dominates if it covers more time than black
                snow_dur = sum(e - s for s, e in run.snows)
                black_dur = sum(e - s for s, e in run.blacks) if run.blacks else 0
                kind = "snow" if snow_dur >= black_dur else "black"
            elif run.whites and not run.blacks:
                kind = "white"
            else:
                kind = "black"
        else:
            # static mode: any signal qualifies
            kind = "mixed"

        dead_spans.append(DeadSpan(start=run.start, end=run.end, kind=kind))

    return sorted(dead_spans, key=lambda d: d.start)


def analyze_full_video(input_video=None, *, from_log=None, mode="black",
                       merge_gap=2.0, min_dead_dur=3.0, min_content_dur=5.0,
                       **probe_kwargs) -> FullVideoAnalysis:
    """Probe the ENTIRE video for dead space and return all dead + content spans.

    Unlike analyze() which only proposes a head/tail trim, this function finds
    dead spans throughout the video and returns the alternating dead/content
    regions that cover the full duration.

    Args:
        input_video:     path to video file
        from_log:        re-score from saved probe log
        mode:            "black" or "static"
        merge_gap:       bridge dead regions closer than this (s)
        min_dead_dur:    minimum dead span duration to report (s)
        min_content_dur: content spans shorter than this get absorbed into adjacent dead
        **probe_kwargs:  forwarded to probe_video()

    Returns:
        FullVideoAnalysis with dead_spans and content_spans covering [0, duration].
    """
    if from_log:
        duration, regions = parse_log(from_log)
    else:
        if not input_video:
            raise ValueError("analyze_full_video() needs input_video or from_log")
        duration, regions = probe_video(input_video, **probe_kwargs)

    dead_spans = find_all_dead_spans(regions, duration, merge_gap=merge_gap,
                                     min_dead_dur=min_dead_dur, mode=mode)

    # Build content spans from the gaps between dead spans
    content_spans = []
    notes = []
    prev_end = 0.0

    for ds in dead_spans:
        if ds.start > prev_end + 0.1:  # content gap before this dead span
            content_spans.append(ContentSpan(start=prev_end, end=ds.start))
        prev_end = ds.end

    # Trailing content after last dead span
    if prev_end < duration - 0.1:
        content_spans.append(ContentSpan(start=prev_end, end=duration))

    # If no dead spans found, the entire video is one content span
    if not dead_spans and not content_spans:
        content_spans.append(ContentSpan(start=0.0, end=duration))

    # Filter out very short content spans (absorb into surrounding dead)
    filtered_content = []
    for cs in content_spans:
        if cs.end - cs.start >= min_content_dur:
            filtered_content.append(cs)
        else:
            notes.append(f"short content span {cs.start:.1f}-{cs.end:.1f}s "
                         f"({cs.end - cs.start:.1f}s) absorbed into dead space")
            # Extend the nearest dead span to cover it
            for ds in dead_spans:
                if abs(ds.end - cs.start) < 0.5:
                    ds.end = cs.end
                    break
                elif abs(ds.start - cs.end) < 0.5:
                    ds.start = cs.start
                    break

    content_spans = filtered_content

    # Determine status
    status = "OK"
    total_content = sum(cs.end - cs.start for cs in content_spans)
    if total_content <= 0:
        status = "NEEDS_REVIEW"
        notes.append("no content spans found (entire video appears dead)")
    elif total_content / duration < 0.1:
        status = "NEEDS_REVIEW"
        notes.append(f"only {100 * total_content / duration:.0f}% content")

    return FullVideoAnalysis(
        duration=duration,
        dead_spans=dead_spans,
        content_spans=content_spans,
        status=status,
        notes=notes,
        regions=regions,
    )


# ------------------------------------ main ------------------------------------
def main():
    p = argparse.ArgumentParser(description="Probe head/tail dead space in a video.")
    p.add_argument("input_video", nargs="?", help="Local video file (omit when using --from-log).")
    p.add_argument("--from-log", help="Re-score a saved probe .txt offline (no ffmpeg).")
    p.add_argument("-o", "--output", default="trimmed.mp4")
    p.add_argument("--apply", action="store_true", help="Write the trimmed content window.")
    p.add_argument("--force", action="store_true", help="Apply even if status is NEEDS_REVIEW.")
    p.add_argument("--mode", choices=["black", "static"], default="black")
    p.add_argument("--no-black", action="store_true")
    p.add_argument("--no-white", action="store_true", help="Skip the white-frame detector.")
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--no-silence", action="store_true")
    p.add_argument("--no-bars", action="store_true", help="Skip the color-bar detector (opencv).")
    p.add_argument("--bars-window", type=float, default=150.0,
                   help="Seconds from the start to scan for a color-bar leader.")
    p.add_argument("--no-snow", action="store_true", help="Skip the video-static/snow detector (opencv).")
    p.add_argument("--snow-window", type=float, default=150.0,
                   help="Seconds at each edge to scan for video static/snow.")
    p.add_argument("--min-dur", type=float, default=0.5)
    p.add_argument("--silence-db", type=float, default=-30.0)
    p.add_argument("--freeze-db", type=float, default=-30.0,
                   help="freezedetect noise tolerance (dB). -30 suits noisy/analog sources.")
    p.add_argument("--black-pic-th", type=float, default=0.98)
    p.add_argument("--merge-gap", type=float, default=0.5)
    p.add_argument("--edge-tol", type=float, default=5.0,
                   help="A dead run starting/ending within this of an edge counts as head/tail. "
                        "~5s tolerates the unstable signal that often precedes a bars leader.")
    p.add_argument("--min-keep", type=float, default=0.5,
                   help="Flag NEEDS_REVIEW if kept span < this fraction of the file.")
    args = p.parse_args()

    if not args.from_log and not args.input_video:
        p.error("provide a video file or --from-log")

    # analyze() ignores the detector kwargs when from_log is set, so we can pass
    # everything unconditionally.
    prop = analyze(
        input_video=args.input_video, from_log=args.from_log, mode=args.mode,
        merge_gap=args.merge_gap, edge_tol=args.edge_tol, min_keep=args.min_keep,
        min_dur=args.min_dur, black_pic_th=args.black_pic_th, freeze_db=args.freeze_db,
        silence_db=args.silence_db, bars_window=args.bars_window, snow_window=args.snow_window,
        no_black=args.no_black, no_white=args.no_white,
        no_freeze=args.no_freeze, no_silence=args.no_silence,
        no_bars=args.no_bars, no_snow=args.no_snow,
    )

    name = args.from_log or args.input_video
    print(f"Input: {name}  (duration {prop.duration:.2f}s)")
    if not args.from_log:
        print("Detected dead regions:")
        for r in sorted(prop.regions, key=lambda x: (x.start, x.kind)):
            print(f"  [{r.start:8.2f} -> {r.end:8.2f}]  {r.kind}")

    print(f"\n[mode={args.mode}] content {prop.content_start:.2f}s -> {prop.content_end:.2f}s | "
          f"trims {prop.content_start:.2f}s head, {prop.tail_trim:.2f}s tail | "
          f"kept {prop.kept:.1f}s ({prop.kept_pct:.0f}%) | status: {prop.status}")
    for n in prop.notes:
        print(f"  note: {n}")

    if args.apply:
        if args.from_log:
            print("Cannot --apply in --from-log mode (no video).")
            return
        if prop.status == "NEEDS_REVIEW" and not args.force:
            print("Refusing to apply: status is NEEDS_REVIEW (use --force to override).")
            return
        if prop.kept <= 0:
            print("Nothing to write (empty span).")
            return
        print(f"Writing trimmed content to: {args.output}")
        apply_trim(args.input_video, args.output, prop.content_start, prop.content_end)
        print("Done.")


if __name__ == "__main__":
    main()
