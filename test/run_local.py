#!/usr/bin/env python3
"""Local, Bedrock-free pipeline runner for the DETERMINISTIC stages.

Produces a `segment/<stem>.json` from a video by running ONLY the parts that don't
need AWS: full-video dead-space detection (ffmpeg + OpenCV) and program grouping.
The LLM stages (Transcribe, Bedrock boundary/label passes) are SKIPPED — so the
output is dead-space-only structure (head/tail + mid-video D spans, bars, snow),
which is exactly what the deterministic work (dead-space thresholds, splice/bars/
snow detection, the carve, program grouping) changes.

This is the FAST inner loop: no cdk deploy, no ECS, no 10-RPM Bedrock cap. Run it
against test/video/*.mp4, then score with regression_harness.py.

What it does NOT do: content segmentation, labels, titles, descriptions, transcripts
(those need Transcribe + Bedrock). So `type-accuracy` / content-`boundaries` in the
harness will look poor here — that's expected; use this to iterate on DEAD-SPACE and
program-structure metrics, and the full cloud path for label/boundary quality.

Requires: ffmpeg on PATH, `pip install opencv-python-headless numpy`.

Usage:
    python run_local.py test/video/spc-mss1159-s02-i0100.mp4            # -> ./out/<stem>.json
    python run_local.py test/video/*.mp4 -o /tmp/out                    # batch
    python run_local.py --video-dir test/video -o /tmp/out              # whole dir
    # then:
    python regression_harness.py --produced-dir /tmp/out --truth-dir test/segment
"""

import argparse
import glob
import json
import os
import sys

# Import the pipeline module. It lives in ../video-processing.
HERE = os.path.dirname(os.path.abspath(__file__))
VP = os.path.join(os.path.dirname(HERE), "video-processing")
sys.path.insert(0, VP)

# main.py imports boto3 at module load, but the DETERMINISTIC path we run never calls
# AWS. Stub boto3/botocore if they're absent so this runner stays dependency-light
# (only ffmpeg + opencv + numpy are truly needed).
import types
for _name in ("boto3",):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)
try:
    import botocore.exceptions  # noqa: F401
except ImportError:
    _bc = types.ModuleType("botocore")
    _ex = types.ModuleType("botocore.exceptions")
    class _ClientError(Exception):
        pass
    _ex.ClientError = _ClientError
    sys.modules["botocore"] = _bc
    sys.modules["botocore.exceptions"] = _ex


def _load_turns(transcript_dir, stem):
    """Load cached diarized turns [{start,end,speaker,text}] for `stem`, or None.
    Only the .json form is usable (the pipeline needs timestamps + speakers); .txt
    transcripts are ignored here (reserved for a later verification step)."""
    path = os.path.join(transcript_dir, f"{stem}.json")
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
        turns = data if isinstance(data, list) else data.get("turns", [])
        # Validate shape minimally.
        return [t for t in turns if {"start", "end", "speaker", "text"} <= set(t)]
    except Exception:
        return None


def _dead_and_prop(video_path, turns):
    """Full-video dead-space analysis -> (prop, mid_dead_spans), mirroring run_detect.
    Uses the transcript (if any) for the speech-veto so talky regions aren't
    over-trimmed. Returns (prop, mid_dead_spans, fa)."""
    import main
    from analyze_deadspace import analyze_full_video, Proposal
    from dataclasses import replace

    speech_windows = []
    if turns:
        # Reuse the same helper the pipeline uses to build speech windows from turns.
        speech_windows = main.speech_windows_from_turns(turns)

    fa = analyze_full_video(video_path, mode=getattr(main, "TRIM_MODE", "black"),
                            speech_windows=speech_windows)
    if fa.content_spans:
        cs_start, cs_end = fa.content_spans[0].start, fa.content_spans[-1].end
    else:
        cs_start, cs_end = 0.0, fa.duration
    prop = Proposal(fa.duration, cs_start, cs_end, fa.status, fa.notes, fa.regions)

    lead_end = main.head_leader_end(fa.regions)
    if lead_end > cs_start + 0.5:
        prop = replace(prop, content_start=lead_end)
        cs_start = lead_end

    mid_dead_spans = [ds for ds in fa.dead_spans
                      if ds.start > cs_start + 0.5 and ds.end < cs_end - 0.5]
    return prop, mid_dead_spans, fa


def _dead_only_json(video_path, turns=None):
    """Dead-space-only segment JSON (no content segmentation at all)."""
    import main
    prop, mid_dead_spans, _ = _dead_and_prop(video_path, turns)
    key = f"video/{os.path.basename(video_path)}"
    doc = main.build_segment_json(key, prop, content_segments=None,
                                  taxonomy=None, turns=None,
                                  mid_dead_spans=mid_dead_spans)
    doc["labels_attempted"] = False   # no content labels in dead-only mode
    return doc


def _fusion_no_bedrock_json(video_path, turns):
    """Transcript-driven segmentation WITHOUT Bedrock.

    Runs the deterministic fusion stages that only need the transcript + shots:
    shot detection, speaker-continuity merge, lexical + pause boundaries (NO LLM
    analyze_program boundaries), topic split, role/guest tagging, boundary
    confidence, and conversation coalesce. Segment TYPES/labels are left as a
    neutral placeholder because those come from label_segments (Bedrock). So this
    lights up `boundaries` and program-grouping locally; `type-accuracy` still needs
    the cloud loop. Mirrors run_fusion_segmentation minus the two LLM calls."""
    import main
    from segment_shots import detect_shots
    from segment_fuse import attach, merge_by_speaker
    from segment_label import (detect_host, segment_text, split_on_topics,
                               lexical_boundaries, dominant_role, segment_guests,
                               pause_boundaries, boundary_confidence,
                               coalesce_conversation)

    prop, mid_dead_spans, fa = _dead_and_prop(video_path, turns)

    window = (prop.content_start, prop.content_end)
    shots = detect_shots(video_path, window,
                         getattr(main, "SEGMENT_DETECTOR", "adaptive"),
                         getattr(main, "SEGMENT_MIN_SHOT", 1.5))
    fused = attach(shots, [(t["start"], t["end"], t["speaker"], t["text"]) for t in turns])

    host = detect_host(turns)
    # NO analyze_program (that's the Bedrock boundary pass). Use only the free cues.
    lex_bounds = lexical_boundaries(turns)
    pause_bounds = pause_boundaries(turns)
    dead_spans = fa.dead_spans
    dead_bounds = []
    for ds in dead_spans:
        dead_bounds += [round(ds.start, 2), round(ds.end, 2)]
    boundaries = sorted(set(lex_bounds) | set(pause_bounds) | set(dead_bounds))

    segments = merge_by_speaker(fused, host=host)
    segments = split_on_topics(segments, boundaries, fused)
    for s in segments:
        s["text"], s["speakers"] = segment_text(turns, s["start"], s["end"])
        s["role"] = dominant_role(turns, s["start"], s["end"], host)
        s["guests"] = segment_guests(turns, s["start"], s["end"], host)
        # No Bedrock label — leave a neutral placeholder the harness treats as content.
        s["label"] = "other"
        s["name"] = ""
        s["description"] = ""
        s["on_screen_text"] = ""

    labeled = coalesce_conversation(segments, host=host, dead_spans=dead_spans)
    shot_starts = sorted({s["start"] for s in fused})
    labeled = boundary_confidence(labeled, turns, shot_starts, [], lex_bounds, host)

    key = f"video/{os.path.basename(video_path)}"
    doc = main.build_segment_json(key, prop, content_segments=labeled,
                                  taxonomy=None, turns=turns,
                                  mid_dead_spans=mid_dead_spans)
    # Segment TYPES are placeholders here (labeling needs Bedrock). Mark that so the
    # harness scores type-accuracy as n/a instead of failing on unlabeled content.
    doc["labels_attempted"] = False
    return doc


def main_cli():
    ap = argparse.ArgumentParser(description="Local Bedrock-free deterministic pipeline runner.")
    ap.add_argument("videos", nargs="*", help="Video file(s) to process.")
    ap.add_argument("--video-dir", help="Process every *.mp4 in this directory.")
    ap.add_argument("-o", "--out-dir", default=os.path.join(HERE, "out"),
                    help="Where to write <stem>.json (default: test/out).")
    ap.add_argument("--transcript-dir", default=os.path.join(HERE, "transcript"),
                    help="Dir of cached diarized transcript JSONs (default: test/transcript).")
    ap.add_argument("--dead-only", action="store_true",
                    help="Force dead-space-only mode even when a transcript exists.")
    args = ap.parse_args()

    videos = list(args.videos)
    if args.video_dir:
        videos += sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not videos:
        ap.print_help()
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    rc = 0
    for v in videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        turns = None if args.dead_only else _load_turns(args.transcript_dir, stem)
        mode = "fusion(no-bedrock)" if turns else "dead-only"
        try:
            doc = _fusion_no_bedrock_json(v, turns) if turns else _dead_only_json(v)
            out_path = os.path.join(args.out_dir, f"{stem}.json")
            with open(out_path, "w") as f:
                json.dump(doc, f, indent=2)
            segs = doc.get("segments", [])
            n_dead = sum(1 for s in segs if s.get("segment_type") == "D")
            n_prog = len(doc.get("programs", []))
            print(f"OK  {stem} [{mode}]: {len(segs)} segments "
                  f"({n_dead} dead, {len(segs) - n_dead} content, {n_prog} programs) -> {out_path}")
        except Exception as e:
            import traceback
            print(f"ERR {stem} [{mode}]: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main_cli())
