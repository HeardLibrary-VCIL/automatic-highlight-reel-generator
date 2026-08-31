"""ECS entry point: download video from S3, detect dead space, segment content, write results.

Integrated with SCUA-Video-Editing Amplify frontend:
  - Input  : S3_BUCKET / S3_KEY pointing at uploaded video under `video/` prefix
  - Output : segment JSON  → `segment/{basename}.json`  (for the Editor timeline)
             review marker → `review/{basename}.txt`    (if NEEDS_REVIEW)

Pipeline stages:
  1. Dead-space detection (analyze_deadspace): find head/tail dead regions
  2. Audio-visual fusion segmentation:
       A. transcribe   -- Amazon Transcribe reads the video straight from S3
                          (already uploaded by the frontend) -> diarized turns
       B. fuse         -- PySceneDetect shot cuts + join words/speakers onto each
                          shot, then merge shots by speaker continuity
       C. segment/label-- boundaries from an LLM pass + lexical (TextTiling) + pause
                          cues; split, recompute host/guest role, then ONE multimodal
                          Bedrock call labels each segment from its words AND a frame
                          (also reading any on-screen caption)
       D. coalesce     -- merge by speaker continuity so one interview stays one
                          segment across camera cuts; score each boundary's confidence
  3. Write combined segment JSON to S3 for the Editor UI

All model calls go through Amazon Bedrock using the ECS task role, so no
ANTHROPIC_API_KEY is needed anywhere in the deployed stack.

The actual video trimming is triggered later by the user from the Editor UI,
after they review and adjust the auto-detected segments.
"""
import json
import os
import re
import subprocess
import sys
import logging
import tempfile
import traceback
import time
import uuid
from dataclasses import replace
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from analyze_deadspace import analyze, analyze_full_video, apply_trim

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Output prefixes — match SCUA Amplify storage paths
SEGMENT_PREFIX = os.environ.get("SEGMENT_PREFIX", "segment").strip("/")
TRIM_MODE = os.environ.get("TRIM_MODE", "black")
# Content-type segmentation. Credentials come from the ECS task role (Bedrock +
# Transcribe), so "auto" simply means on; set CONTENT_SEGMENT=off to get
# dead-space-only segments.
CONTENT_SEGMENT = os.environ.get("CONTENT_SEGMENT", "auto")  # "auto", "on", "off"
SEGMENT_DETECTOR = os.environ.get("SEGMENT_DETECTOR", "adaptive")  # "adaptive" or "content"
SEGMENT_MIN_SHOT = float(os.environ.get("SEGMENT_MIN_SHOT", "1.5"))
TRANSCRIBE_LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "en-US")
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "10"))
# Region for boto3 service clients. The container sets AWS_REGION, but botocore
# resolves from AWS_DEFAULT_REGION -- and Transcribe (unlike S3) has no global
# fallback, so we pass region_name explicitly rather than rely on either var.
REGION = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
          or "us-east-1")
# Full-video dead-space detection tuning
MERGE_GAP = float(os.environ.get("MERGE_GAP", "2.0"))
MIN_DEAD_DUR = float(os.environ.get("MIN_DEAD_DUR", "3.0"))
MIN_CONTENT_DUR = float(os.environ.get("MIN_CONTENT_DUR", "5.0"))
BLACK_PIC_TH = float(os.environ.get("BLACK_PIC_TH", "0.995"))
MAX_FRAME_WIDTH = int(os.environ.get("MAX_FRAME_WIDTH", "768"))
# Set FULL_VIDEO_SCAN=on to detect dead space throughout (not just head/tail)
FULL_VIDEO_SCAN = os.environ.get("FULL_VIDEO_SCAN", "on").lower() in ("on", "true", "1")


def segment_key(s3_key: str) -> str:
    """Map an input key to its segment JSON key under segment/."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    return f"{SEGMENT_PREFIX}/{stem}.json"


def safe_output_name(name: str, fallback_stem: str) -> str:
    """An S3-safe basename (with a single .mp4) for the trimmed clip. Honors the
    reviewer's chosen name from the trim request but strips path parts and unsafe
    chars; falls back to '{fallback_stem}_trimmed.mp4' when no usable name is given."""
    base = Path((name or "").strip()).name
    base = re.sub(r"\.mp4$", "", base, flags=re.IGNORECASE).strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "", base).strip()
    return f"{base}.mp4" if base else f"{fallback_stem}_trimmed.mp4"


def _should_run_content_segmentation() -> bool:
    """Whether to run the fusion segmentation. Bedrock/Transcribe auth comes from
    the ECS task role, so there is no key to check -- only the explicit off switch."""
    return CONTENT_SEGMENT != "off"


def turns_to_vtt(turns: list) -> str:
    """Convert speaker turns to WebVTT format (for Aviary/captioning)."""
    lines = ["WEBVTT", ""]
    for i, t in enumerate(turns, 1):
        sh, sm, ss = int(t["start"] // 3600), int((t["start"] % 3600) // 60), t["start"] % 60
        eh, em, es = int(t["end"] // 3600), int((t["end"] % 3600) // 60), t["end"] % 60
        lines.append(str(i))
        lines.append(f"{sh:02d}:{sm:02d}:{ss:06.3f} --> {eh:02d}:{em:02d}:{es:06.3f}")
        speaker = t.get("speaker", "")
        text = t.get("text", "")
        lines.append(f"<v {speaker}>{text}" if speaker and speaker != "spk_?" else text)
        lines.append("")
    return "\n".join(lines)


def extract_audio(input_video: str, output_audio: str) -> str:
    """Extract the audio track to an mp4/aac clip for Amazon Transcribe, used when the
    uploaded container isn't one Transcribe reads directly (avi/mkv/wmv/ts/...). Raises
    if the source has no audio stream, which the caller treats as "no usable audio" and
    drops to visual segmentation."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", input_video, "-vn", "-c:a", "aac",
         "-b:a", "128k", output_audio],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=True,
    )
    return output_audio


def has_audio_stream(input_video: str) -> bool:
    """True if the video has at least one audio stream. Silent videos (e.g. a
    game clip with no commentary) have none — we skip transcription for those
    and segment visually, avoiding a wasted/failing Transcribe cycle."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", input_video],
            stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return False


def get_transcript_turns(local_video_path, s3_bucket, s3_key):
    """Transcribe the video (or load cached transcript). Returns diarized turns:
    a list of {"start","end","speaker","text"}. Cached to transcript/{stem}.json so
    it only runs Transcribe once even though both the dead-space pre-check and the
    fusion segmentation need it. Returns [] if no usable audio/speech."""
    import boto3 as _boto3
    from transcribe import start_job, wait, fetch_result, to_speaker_turns, media_format

    stem = Path(s3_key).name.rsplit(".", 1)[0]
    s3_client = _boto3.client("s3", region_name=REGION)
    transcript_cache_key = f"transcript/{stem}.json"

    # Cache check
    try:
        resp = s3_client.get_object(Bucket=s3_bucket, Key=transcript_cache_key)
        turns = json.loads(resp["Body"].read().decode("utf-8"))
        if turns:
            log.info(f"[A] Loaded cached transcript ({len(turns)} turns)")
            return turns
    except Exception:
        pass

    transcribe_client = _boto3.client("transcribe", region_name=REGION)
    job_name = f"scua-{stem}-{int(time.time())}-{uuid.uuid4().hex[:8]}"[:200]
    media_uri = f"s3://{s3_bucket}/{s3_key}"
    if media_format(s3_key) is None:
        audio_key = f"edit/{stem}.transcribe.mp4"
        audio_path = Path(local_video_path).with_name(f"{stem}.transcribe.mp4")
        extract_audio(str(local_video_path), str(audio_path))
        s3_client.upload_file(str(audio_path), s3_bucket, audio_key)
        media_uri = f"s3://{s3_bucket}/{audio_key}"
        log.info(f"[A] extracted audio -> s3://{s3_bucket}/{audio_key}")
    log.info(f"[A] Transcribe job {job_name} on {media_uri}")
    start_job(transcribe_client, media_uri, job_name=job_name,
              language=TRANSCRIBE_LANGUAGE, max_speakers=MAX_SPEAKERS)
    job = wait(transcribe_client, job_name)
    turns = to_speaker_turns(fetch_result(job, s3_client))
    if not turns:
        return []
    # Cache transcript + VTT
    try:
        s3_client.put_object(Bucket=s3_bucket, Key=transcript_cache_key,
                             Body=json.dumps(turns, indent=2).encode("utf-8"),
                             ContentType="application/json")
        s3_client.put_object(Bucket=s3_bucket,
                             Key=transcript_cache_key.replace(".json", ".vtt"),
                             Body=turns_to_vtt(turns).encode("utf-8"),
                             ContentType="text/vtt")
        log.info(f"[A] Cached transcript + VTT")
    except Exception as cache_err:
        log.warning(f"[A] Could not cache transcript: {cache_err}")
    return turns


def speech_windows_from_turns(turns, merge_gap=5.0):
    """Collapse diarized turns into merged speech time windows [(start,end)].
    Adjacent turns within merge_gap seconds are joined so brief pauses between
    sentences don't fragment the windows.

    merge_gap is deliberately small (5s): a longer gap between turns means a real
    silent stretch (e.g. a black-screen filler or bars leader between segments),
    which must NOT be absorbed into a speech window — otherwise the dead-space
    veto would hide those mid-video dead spans. Only bridge true sentence pauses."""
    if not turns:
        return []
    spans = sorted((float(t["start"]), float(t["end"])) for t in turns)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def run_fusion_segmentation(local_video_path, s3_bucket, s3_key, prop, turns=None):
    """Stages A-D: transcribe -> shots+fuse -> discover taxonomy/topics -> label.

    Transcribe reads the ORIGINAL S3 object (the frontend already uploaded it), so
    nothing is re-uploaded; the local copy is only used for shot detection and for
    sampling one frame per segment. Returns (labeled_segments, taxonomy, turns) --
    the diarized `turns` are handed back so the segment JSON can carry each
    segment's transcript for the reviewer.
    """
    import boto3 as _boto3
    from transcribe import start_job, wait, fetch_result, to_speaker_turns, media_format
    from segment_shots import detect_shots
    from segment_fuse import attach, merge_by_speaker
    from segment_label import (analyze_program, detect_host, segment_text,
                               split_on_topics, label_segments, coalesce_conversation,
                               lexical_boundaries, dominant_role, segment_guests,
                               pause_boundaries, boundary_confidence)
    from bedrock import make_client

    stem = Path(s3_key).name.rsplit(".", 1)[0]
    import boto3 as _boto3
    s3_client = _boto3.client("s3", region_name=REGION)

    # --- A. transcribe straight from S3 (diarized) — reuse pre-fetched turns ---
    if turns is None:
        turns = get_transcript_turns(local_video_path, s3_bucket, s3_key)
    if not turns:
        raise RuntimeError("Transcribe returned no speech turns")

    # --- B. shots + join words onto them ---
    t0 = time.time()
    window = (prop.content_start, prop.content_end)
    shots = detect_shots(str(local_video_path), window, SEGMENT_DETECTOR, SEGMENT_MIN_SHOT)
    fused = attach(shots, [(t["start"], t["end"], t["speaker"], t["text"]) for t in turns])
    log.info(f"[B] {len(fused)} shots in window {window[0]:.1f}-{window[1]:.1f}s "
             f"({time.time() - t0:.1f}s)")

    # --- C. discover taxonomy + multi-cue boundaries, split, then label multimodally ---
    t0 = time.time()
    client = make_client()
    host = detect_host(turns)
    taxonomy, llm_bounds = analyze_program(turns, client)
    # Two more cue sources, both free from data we already have: TextTiling lexical
    # valleys catch mid-shot boundaries (a host wrap-up rolling into a PSA with no
    # camera cut); long silences between turns mark broadcast segment breaks. The
    # speaker-continuity coalesce removes any that fall inside one conversation.
    lex_bounds = lexical_boundaries(turns)
    pause_bounds = pause_boundaries(turns)
    boundaries = sorted(set(llm_bounds) | set(lex_bounds) | set(pause_bounds))
    log.info(f"[C] host={host} | taxonomy: " + ", ".join(t["type"] for t in taxonomy)
             + f" | {len(boundaries)} boundaries (LLM {len(llm_bounds)} + lexical "
             f"{len(lex_bounds)} + pause {len(pause_bounds)})")

    segments = merge_by_speaker(fused, host=host)
    segments = split_on_topics(segments, boundaries, fused)
    for s in segments:
        s["text"], s["speakers"] = segment_text(turns, s["start"], s["end"])
        # Re-tag role by who actually dominates this (possibly newly split) span, so a
        # host wrap-up sliced off a guest segment reads as host, not the guest's story.
        s["role"] = dominant_role(turns, s["start"], s["end"], host)
        s["guests"] = segment_guests(turns, s["start"], s["end"], host)

    labeled = label_segments(segments, client, taxonomy, str(local_video_path), host=host)
    # --- D. coalesce by conversation continuity (guest speaker), not visual label:
    # a continuous interview stays ONE segment even when the camera keeps cutting.
    labeled = coalesce_conversation(labeled, host=host)
    # Score each final boundary by how many independent cues agree, for reviewer triage.
    shot_starts = sorted({s["start"] for s in fused})
    labeled = boundary_confidence(labeled, turns, shot_starts, llm_bounds, lex_bounds, host)
    log.info(f"[C/D] {len(segments)} -> {len(labeled)} labeled segments "
             f"({time.time() - t0:.1f}s)")
    return labeled, taxonomy, turns


def run_visual_segmentation(local_video_path, prop):
    """Segment a video that has NO usable audio by CONTENT TYPE — the standard
    two-level approach (shot-boundary detection -> group shots into scenes) rather
    than fixed-interval chopping:

        PySceneDetect shots -> classify each shot from a frame with Claude/Bedrock
        (dance performance / football game / interview / political ad / ... / other,
        plus a short free-text description) -> merge ADJACENT same-type shots into
        one scene -> name each scene by its content type.

    So segments read as "Interview", "Football game", "Commercial", etc. instead of
    generic "Scene N", and the boundaries fall where the CONTENT actually changes.
    No transcript needed, so this covers the silent / no-speech case.

    Honest limit: a video that is one uniform type throughout (e.g. a raw fixed-
    camera game) is correctly ONE scene — fine-grained sub-segmentation of a single
    type (e.g. per-play) needs a domain signal like scoreboard OCR, not implemented.
    Raises if nothing could be classified so the caller can drop to dead-space only.
    """
    from segment_shots import detect_shots, label_shots, _shots_to_segments
    from segment_content import (confirm_targets, smooth_labels, consolidate_segments,
                                  CATEGORIES, OTHER, other_note)

    window = (prop.content_start, prop.content_end)
    shots = detect_shots(str(local_video_path), window, SEGMENT_DETECTOR, SEGMENT_MIN_SHOT)

    # One Bedrock vision call per shot -> (shot_start, content-type, short description).
    labeled = label_shots(str(local_video_path), shots)
    kept = {round(ss, 2) for ss, _, _ in labeled}
    shots = [(ss, ee) for ss, ee in shots if round(ss, 2) in kept]
    if not shots:
        raise RuntimeError("visual classification produced no labelled shots")

    cats = [(ss, c) for ss, c, _ in labeled]
    cats = smooth_labels(confirm_targets(cats, CATEGORIES, min_windows=1))
    segments = _shots_to_segments(shots, cats)      # merge adjacent same-type shots
    segments = consolidate_segments(segments, 0, 0.0)
    for s in segments:
        if s.label == OTHER:
            s.note = other_note(s, labeled)          # a description to name the 'other' span

    log.info(f"Visual content-type segmentation: {len(labeled)} shots -> {len(segments)} scenes")
    for s in segments:
        note = getattr(s, "note", "")
        log.info(f"  [{s.start:.1f}s -> {s.end:.1f}s] {s.label}"
                 + (f" ({note})" if s.label == OTHER and note else ""))
    return [{
        "start": round(s.start, 2),
        "end": round(s.end, 2),
        # Known content type -> 'C' segment titled by the type; 'other' -> 'I'
        # segment titled by the model's short description (build_segment_json).
        "label": s.label,
        "name": (getattr(s, "note", "") or "") if s.label == OTHER else "",
        "speakers": [],
        "on_screen_text": "",
        "confidence": "",
        "cues": ["shot", "vlm"],
    } for s in segments]


def build_segment_json(s3_key: str, prop, content_segments=None, taxonomy=None,
                       turns=None, mid_dead_spans=None) -> dict:
    """Build a segment JSON matching SCUA Editor format.

    `mid_dead_spans` are DeadSpan objects that fall BETWEEN content (e.g. a color
    bars leader or black gap mid-program). They're inserted as "D" segments so the
    Editor shows them; without this they'd be swallowed by the content window.

    `content_segments` are the fusion-pipeline dicts
    {start, end, label, name, speakers}; the Editor gets the specific `name` as
    the title (e.g. "Triangle Inn resort tour") and the content type as
    `content_label`. Without them we fall back to simple D/I dead-space segments.

    `turns` are the diarized transcript rows; when present, each content segment
    carries a `transcript` of the words spoken over it so a reviewer can verify
    the label and boundaries against what is actually said.
    """
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    segments = []

    # Recompute transcript from `turns` against each FINAL segment span rather than
    # trusting the seg["text"] carried on the dict: that text predates coalescing
    # and would be truncated to the first merged child.
    if turns:
        from segment_label import segment_text

        def _transcript(start, end):
            return segment_text(turns, start, end)[0]
    else:
        def _transcript(start, end):
            return ""

    if content_segments:
        # Use the fusion segmentation results, keeping head/tail dead-space markers
        if prop.content_start > 1.0:
            segments.append({
                "segment_start": 0,
                "segment_end": round(prop.content_start, 2),
                "segment_type": "D",
                "title": "Head dead space (auto-detected)",
            })

        for seg in content_segments:
            label = seg.get("label", "other")
            name = (seg.get("name") or "").strip()
            # Editor segment_type codes: "C" = a recognized content type,
            # "I" = generic/unclassified content.
            seg_type = "I" if label == "other" else "C"
            segments.append({
                "segment_start": round(seg["start"], 2),
                "segment_end": round(seg["end"], 2),
                "segment_type": seg_type,
                "title": name or label.title(),
                "content_label": label,
                "speakers": seg.get("speakers", []),
                "transcript": _transcript(seg["start"], seg["end"]),
                # On-screen caption (name/title read off the frame) + how much the
                # boundary before this segment is trusted, for reviewer triage.
                "caption": seg.get("on_screen_text", ""),
                "confidence": seg.get("confidence", ""),
                "boundary_cues": seg.get("cues", []),
            })

        if prop.duration - prop.content_end > 1.0:
            segments.append({
                "segment_start": round(prop.content_end, 2),
                "segment_end": round(prop.duration, 2),
                "segment_type": "D",
                "title": "Tail dead space (auto-detected)",
            })
    else:
        # Fallback: simple dead-space-only segments
        if prop.content_start > 1.0:
            segments.append({
                "segment_start": 0,
                "segment_end": round(prop.content_start, 2),
                "segment_type": "D",
                "title": "Head dead space (auto-detected)",
            })

        segments.append({
            "segment_start": round(prop.content_start, 2),
            "segment_end": round(prop.content_end, 2),
            "segment_type": "I",
            "title": "Content",
        })

        if prop.duration - prop.content_end > 1.0:
            segments.append({
                "segment_start": round(prop.content_end, 2),
                "segment_end": round(prop.duration, 2),
                "segment_type": "D",
                "title": "Tail dead space (auto-detected)",
            })

    # Inject mid-video dead spans (color bars leaders, black gaps between segments).
    # These fall inside the content window, so they'd otherwise be hidden. For each
    # mid dead span, carve it out of any overlapping content segment and add a "D"
    # marker, then re-sort. Head/tail spans (at the very edges) are already handled
    # above, so skip anything touching 0 or duration.
    _DEAD_LABELS = {"bars": "Color bars (auto-detected)",
                    "snow": "Video static/snow (auto-detected)",
                    "black": "Black gap (auto-detected)",
                    "white": "White gap (auto-detected)",
                    "mixed": "Dead space (auto-detected)"}
    if mid_dead_spans:
        for ds in mid_dead_spans:
            ds_start, ds_end = round(ds.start, 2), round(ds.end, 2)
            if ds_end - ds_start < 0.5:
                continue
            carved = []
            for seg in segments:
                s0, s1 = seg["segment_start"], seg["segment_end"]
                # No overlap → keep as-is
                if ds_end <= s0 or ds_start >= s1:
                    carved.append(seg)
                    continue
                # Overlap: keep the portion(s) of this segment outside the dead span
                if s0 < ds_start:
                    left = dict(seg); left["segment_end"] = ds_start
                    carved.append(left)
                if s1 > ds_end:
                    right = dict(seg); right["segment_start"] = ds_end
                    carved.append(right)
                # (the middle, [ds_start,ds_end], is replaced by the dead marker below)
            carved.append({
                "segment_start": ds_start,
                "segment_end": ds_end,
                "segment_type": "D",
                "title": _DEAD_LABELS.get(ds.kind, "Dead space (auto-detected)"),
            })
            segments = carved
        # Re-sort and drop any zero/negative-length slivers from carving
        segments = [s for s in segments if s["segment_end"] - s["segment_start"] > 0.05]
        segments.sort(key=lambda s: s["segment_start"])

    # Guarantee the timeline is a gapless partition of [0, duration]: snap the first
    # segment to 0 and the last to the full duration, so a sub-second head/tail sliver
    # below the dead-space-marker threshold isn't left as an uncovered gap that the
    # Editor silently drops when trimming (only listed segments are kept).
    if segments:
        segments[0]["segment_start"] = 0
        segments[-1]["segment_end"] = round(prop.duration, 2)

    out = {
        "video": stem,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auto_detected": True,
        "content_segmented": bool(content_segments),
        "trim_status": prop.status,
        "duration": round(prop.duration, 2),
        "kept_pct": round(prop.kept_pct, 1),
        "segments": segments,
    }
    if taxonomy:
        # The per-video content types the model discovered, so the Editor can show
        # (and let a human correct) the vocabulary the labels came from.
        out["taxonomy"] = taxonomy
    return out


def write_review_marker(s3_client, bucket: str, s3_key: str, prop) -> None:
    """Drop a review marker so a human can find flagged files."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    body = (
        f"source: s3://{bucket}/{s3_key}\n"
        f"status: {prop.status}\n"
        f"proposed content: {prop.content_start:.2f}s -> {prop.content_end:.2f}s\n"
        f"kept: {prop.kept_pct:.0f}%\n"
        + "".join(f"note: {n}\n" for n in prop.notes)
    )
    try:
        s3_client.put_object(Bucket=bucket, Key=f"review/{stem}.txt",
                             Body=body.encode("utf-8"))
    except Exception as e:
        log.warning(f"Could not write review marker: {e}")


# --- Main Orchestrator ---
def main():
    pipeline_start_time = time.time()
    mode = os.environ.get("MODE", "detect")

    s3_bucket = os.environ.get("S3_BUCKET")
    s3_key = os.environ.get("S3_KEY")
    if not s3_bucket or not s3_key:
        log.error("S3_BUCKET and S3_KEY environment variables are required.")
        sys.exit(1)

    if mode == "trim":
        run_trim(s3_bucket, s3_key, pipeline_start_time)
    else:
        run_detect(s3_bucket, s3_key, pipeline_start_time)


def run_trim(s3_bucket, s3_key, pipeline_start_time):
    """Trim video based on user-approved segments from a trim request."""
    log.info("=== SCUA Video Trimmer Starting ===")
    trim_request_key = os.environ.get("TRIM_REQUEST_KEY", "")
    s3_client = boto3.client("s3")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            # Load trim request
            log.info(f"Loading trim request: s3://{s3_bucket}/{trim_request_key}")
            resp = s3_client.get_object(Bucket=s3_bucket, Key=trim_request_key)
            trim_data = json.loads(resp["Body"].read().decode("utf-8"))
            keep_segments = trim_data.get("keep_segments", [])
            video_name = trim_data.get("video", "unknown")

            if not keep_segments:
                log.error("No keep_segments in trim request")
                sys.exit(1)

            # Download video
            filename = Path(s3_key).name
            local_video_path = temp_dir / filename
            log.info(f"Downloading video: s3://{s3_bucket}/{s3_key}")
            s3_client.download_file(s3_bucket, s3_key, str(local_video_path))

            # Trim: concatenate all keep segments
            trimmed_path = temp_dir / f"{video_name}_trimmed.mp4"

            if len(keep_segments) == 1:
                # Single segment — simple cut
                seg = keep_segments[0]
                log.info(f"Trimming single segment: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                apply_trim(str(local_video_path), str(trimmed_path), seg["start"], seg["end"])
            else:
                # Multiple segments — cut each then concatenate
                clip_paths = []
                for i, seg in enumerate(keep_segments):
                    clip_path = temp_dir / f"clip_{i:03d}.mp4"
                    log.info(f"  Cutting clip {i}: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                    apply_trim(str(local_video_path), str(clip_path), seg["start"], seg["end"])
                    clip_paths.append(clip_path)

                # Write concat list
                concat_file = temp_dir / "concat.txt"
                with open(concat_file, "w") as f:
                    for cp in clip_paths:
                        f.write(f"file '{cp}'\n")

                # Concatenate with a RE-ENCODE (not -c copy). The clips are already
                # re-encoded with identical settings, but concatenating with -c copy can
                # still glitch at boundaries (timestamp/GOP discontinuities); re-encoding
                # the joined stream guarantees clean cuts. -c:a aac is a no-op when the
                # source has no audio (silent clips), so this also handles no-audio video.
                log.info(f"Concatenating {len(clip_paths)} clips (re-encode)...")
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_file),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-c:a", "aac", "-b:a", "128k",
                     str(trimmed_path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, check=True,
                )

            # Prefer the explicit output_key from the frontend: the trimmed object
            # is named by the trimmed Video row's permanent id (edit/<id>.mp4), so
            # it's decoupled from any display name and stable across renames. Fall
            # back to the legacy sanitized-name path for older requests.
            output_key = trim_data.get("output_key") or \
                f"edit/{safe_output_name(trim_data.get('output_name'), video_name)}"
            log.info(f"Uploading trimmed video to s3://{s3_bucket}/{output_key}")
            s3_client.upload_file(str(trimmed_path), s3_bucket, output_key)

            log.info(f"=== SCUA Video Trimmer Finished Successfully in "
                     f"{time.time() - pipeline_start_time:.2f}s ===")

        except ClientError as e:
            log.error(f"S3 error: {e.response['Error']['Message']}")
            log.error(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.error(traceback.format_exc())
            sys.exit(1)


def run_detect(s3_bucket, s3_key, pipeline_start_time):
    """Detect dead space and write segment JSON."""
    log.info("=== SCUA Segment Detector Starting ===")
    log.info(f"Processing s3://{s3_bucket}/{s3_key}  (mode={TRIM_MODE})")
    s3_client = boto3.client("s3")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            filename = Path(s3_key).name
            stem = filename.rsplit(".", 1)[0]
            local_video_path = temp_dir / filename
            log.info(f"Downloading video to {local_video_path}...")
            s3_client.download_file(s3_bucket, s3_key, str(local_video_path))

            # --- STAGE 0: TRANSCRIBE FIRST (so speech can veto false dead space) ---
            # VHS audio is low-level and often reads as "silent" to ffmpeg's
            # silencedetect, so we can't rely on audio energy to tell dark program
            # from dead space. The transcript is authoritative: if Transcribe found
            # speech in a range, that range is content regardless of black frames.
            prefetched_turns = None
            speech_windows = []
            video_has_audio = True
            if _should_run_content_segmentation():
                video_has_audio = has_audio_stream(str(local_video_path))
                if not video_has_audio:
                    # Silent video (e.g. a game clip with no commentary): skip
                    # transcription entirely and let Stage 2 segment visually.
                    log.info("--- Stage 0 (Transcribe): no audio stream; "
                             "skipping transcription, will segment visually ---")
                else:
                    try:
                        prefetched_turns = get_transcript_turns(
                            local_video_path, s3_bucket, s3_key)
                        speech_windows = speech_windows_from_turns(prefetched_turns)
                        log.info(f"--- Stage 0 (Transcribe): {len(prefetched_turns or [])} turns, "
                                 f"{len(speech_windows)} speech windows ---")
                    except Exception as e:
                        log.warning(f"Stage 0 transcription failed ({e}); "
                                    f"dead-space runs without speech veto")

            # --- STAGE 1: PROBE ---
            stage1_start = time.time()
            if FULL_VIDEO_SCAN:
                # Full-video dead-space detection: finds ALL dead spans throughout
                full_analysis = analyze_full_video(
                    str(local_video_path), mode=TRIM_MODE,
                    merge_gap=MERGE_GAP, min_dead_dur=MIN_DEAD_DUR,
                    min_content_dur=MIN_CONTENT_DUR, black_pic_th=BLACK_PIC_TH,
                    speech_windows=speech_windows)
                # Build a Proposal-compatible object for the rest of the pipeline
                # using the first content span start and last content span end
                if full_analysis.content_spans:
                    cs_start = full_analysis.content_spans[0].start
                    cs_end = full_analysis.content_spans[-1].end
                else:
                    cs_start, cs_end = 0.0, full_analysis.duration
                from analyze_deadspace import Proposal
                prop = Proposal(full_analysis.duration, cs_start, cs_end,
                                full_analysis.status, full_analysis.notes,
                                full_analysis.regions)
                # Mid-video dead spans (bars/black/snow between content) — these fall
                # inside [cs_start, cs_end] so build_segment_json must inject them as
                # "D" markers, else they're hidden inside the content window.
                mid_dead_spans = [
                    ds for ds in full_analysis.dead_spans
                    if ds.start > cs_start + 0.5 and ds.end < cs_end - 0.5
                ]
                for ds in mid_dead_spans:
                    log.info(f"  MID-DEAD [{ds.start:.1f}s -> {ds.end:.1f}s] {ds.kind}")
                log.info(
                    f"--- Stage 1 (Full-video probe) completed in "
                    f"{time.time() - stage1_start:.2f}s ---\n"
                    f"  Duration: {full_analysis.duration:.1f}s | "
                    f"Dead: {len(full_analysis.dead_spans)} spans | "
                    f"Content: {len(full_analysis.content_spans)} spans | "
                    f"Status: {full_analysis.status}")
                for ds in full_analysis.dead_spans:
                    log.info(f"  DEAD [{ds.start:.1f}s -> {ds.end:.1f}s] {ds.kind}")
            else:
                prop = analyze(str(local_video_path), mode=TRIM_MODE)
                full_analysis = None
                mid_dead_spans = []  # head/tail-only mode has no mid-video dead spans
                log.info(
                    f"--- Stage 1 (Probe) completed in {time.time() - stage1_start:.2f}s --- "
                    f"content {prop.content_start:.2f}s -> {prop.content_end:.2f}s | "
                    f"head {prop.head_trim:.2f}s, tail {prop.tail_trim:.2f}s | "
                    f"kept {prop.kept_pct:.0f}% | status={prop.status}")
            for n in prop.notes:
                log.info(f"  note: {n}")

            # --- STAGE 2: AUDIO-VISUAL FUSION SEGMENTATION ---
            stage2_start = time.time()
            content_segments, taxonomy, transcript_turns = None, None, None

            # Segmentation is decoupled from the dead-space VERDICT: a NEEDS_REVIEW trim
            # (odd leader, short clip, over-eager detectors) should still yield a
            # segmented timeline, not one big block. When the trim is OK we segment the
            # trusted content window; otherwise we segment the WHOLE file and emit no
            # head/tail dead-space markers, since we didn't trust the trim. `seg_prop`
            # drives BOTH the segmentation window and the head/tail markers so they can
            # never disagree; for the OK case it is `prop` itself (no behaviour change).
            seg_prop = (prop if prop.status == "OK" and prop.kept > 0
                        else replace(prop, content_start=0.0, content_end=prop.duration))

            if _should_run_content_segmentation() and seg_prop.kept > 0 and not video_has_audio:
                # Silent video: skip fusion (needs a transcript) and segment by
                # content type visually — labels scenes "Football game", etc.
                log.info("Running visual content-type segmentation (no audio)...")
                try:
                    content_segments = run_visual_segmentation(local_video_path, seg_prop)
                    taxonomy, transcript_turns = None, None
                    log.info(f"Visual segmentation: {len(content_segments)} scenes "
                             f"in {time.time() - stage2_start:.2f}s")
                except Exception as e2:
                    log.warning(f"Visual segmentation failed (falling back to dead-space only): {e2}")
                    log.warning(traceback.format_exc())
                    content_segments, taxonomy, transcript_turns = None, None, None
            elif _should_run_content_segmentation() and seg_prop.kept > 0:
                if prop.status != "OK":
                    log.info(f"Trim status={prop.status}; segmenting whole file "
                             f"0->{prop.duration:.1f}s (no auto dead-space markers)")
                log.info("Running audio-visual fusion segmentation...")
                try:
                    content_segments, taxonomy, transcript_turns = run_fusion_segmentation(
                        local_video_path, s3_bucket, s3_key, seg_prop,
                        turns=prefetched_turns)
                    log.info(f"Fusion segmentation complete: {len(content_segments)} segments "
                             f"in {time.time() - stage2_start:.2f}s")
                    for s in content_segments:
                        log.info(f"  [{s['start']:.2f}s -> {s['end']:.2f}s] "
                                 f"{s['label']} ({s.get('name', '')})")
                    # Cache transcript + VTT to S3 for reuse on re-segmentation
                    if transcript_turns:
                        transcript_s3_key = f"transcript/{stem}.json"
                        try:
                            s3_client.put_object(
                                Bucket=s3_bucket, Key=transcript_s3_key,
                                Body=json.dumps(transcript_turns, indent=2).encode("utf-8"),
                                ContentType="application/json")
                            s3_client.put_object(
                                Bucket=s3_bucket, Key=transcript_s3_key.replace(".json", ".vtt"),
                                Body=turns_to_vtt(transcript_turns).encode("utf-8"),
                                ContentType="text/vtt")
                            log.info(f"  Cached transcript + VTT to s3://{s3_bucket}/{transcript_s3_key}")
                        except Exception as cache_err:
                            log.warning(f"  Could not cache transcript: {cache_err}")
                except Exception as e:
                    # No usable audio (silent clip / no audio stream / Transcribe
                    # failed) or a broken fusion stage: fall back to VISUAL shot
                    # segmentation so the video is still segmented instead of one
                    # big block. Only if THAT fails too do we drop to dead-space only.
                    log.warning(f"Fusion segmentation unavailable ({e}); trying visual shot segmentation")
                    try:
                        content_segments = run_visual_segmentation(local_video_path, seg_prop)
                        taxonomy, transcript_turns = None, None
                        log.info(f"Visual shot segmentation: {len(content_segments)} shots "
                                 f"in {time.time() - stage2_start:.2f}s")
                    except Exception as e2:
                        log.warning(f"Visual segmentation also failed (falling back to dead-space only): {e2}")
                        log.warning(traceback.format_exc())
                        content_segments, taxonomy, transcript_turns = None, None, None
            elif not _should_run_content_segmentation():
                log.info("Content segmentation disabled (CONTENT_SEGMENT=off)")
            else:
                log.info("Skipping content segmentation (empty content span)")

            # --- STAGE 3: WRITE SEGMENTS ---
            stage3_start = time.time()
            seg_s3_key = segment_key(s3_key)

            if prop.status == "NEEDS_REVIEW":
                write_review_marker(s3_client, s3_bucket, s3_key, prop)

            # Always write segment JSON so the Editor can display it. Use `seg_prop`
            # (not `prop`) so the head/tail dead-space markers match the window the
            # segments were actually computed over.
            seg_json = build_segment_json(s3_key, seg_prop, content_segments, taxonomy,
                                          transcript_turns, mid_dead_spans=mid_dead_spans)
            log.info(f"Uploading segment JSON to s3://{s3_bucket}/{seg_s3_key}")
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=seg_s3_key,
                Body=json.dumps(seg_json, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            log.info(f"--- Stage 3 (Write Segments) completed in "
                     f"{time.time() - stage3_start:.2f}s ---")

        except ClientError as e:
            log.error(f"S3 error: {e.response['Error']['Message']}")
            log.error(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.error(traceback.format_exc())
            sys.exit(1)

    log.info(f"=== SCUA Segment Detector Finished Successfully in "
             f"{time.time() - pipeline_start_time:.2f}s ===")


if __name__ == "__main__":
    main()
