"""ECS entry point: download video from S3, detect dead space throughout, label content, write results.

Integrated with SCUA-Video-Editing Amplify frontend:
  - Input  : S3_BUCKET / S3_KEY pointing at uploaded video under `video/` prefix
  - Output : segment JSON  → `segment/{basename}.json`  (for the Editor timeline)
             review marker → `review/{basename}.txt`    (if NEEDS_REVIEW)

Pipeline stages:
  1. Full-video dead-space detection (analyze_deadspace.analyze_full_video):
     Finds ALL dead spans throughout the video (black, white, bars, snow, freeze+silence)
     — not just head/tail. Returns alternating dead spans + content spans.
  2. Transcription (AWS Transcribe):
     Transcribes the full video with speaker diarization. Words are then sliced
     into each content span so archivists can read/edit the transcript per segment.
  3. Content-span labeling (Claude Sonnet via direct Anthropic API):
     For each content span between dead gaps, samples a representative frame
     and sends it to Claude with the segment's transcript for context.
     Returns a short title/description for each content segment.
  4. Write combined segment JSON to S3 for the Editor UI: alternating D and C/I
     segments covering the full video duration, each with transcript text.

The actual video trimming is triggered later by the user from the Editor UI.
"""
import json
import os
import sys
import logging
import tempfile
import traceback
import time
import base64
import cv2
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from analyze_deadspace import analyze_full_video, apply_trim

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
# Content labeling: "auto" = run if ANTHROPIC_API_KEY present; "off" = skip
CONTENT_SEGMENT = os.environ.get("CONTENT_SEGMENT", "auto")
# Transcription
TRANSCRIBE_LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "en-US")
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "10"))
# Dead-space detection tuning
MERGE_GAP = float(os.environ.get("MERGE_GAP", "2.0"))        # bridge dead regions closer than this
MIN_DEAD_DUR = float(os.environ.get("MIN_DEAD_DUR", "3.0"))  # min seconds for a dead span
MIN_CONTENT_DUR = float(os.environ.get("MIN_CONTENT_DUR", "5.0"))  # content spans shorter than this get absorbed
BLACK_PIC_TH = float(os.environ.get("BLACK_PIC_TH", "0.995"))  # black detection strictness (0.98=loose, 0.999=strict)
# Frame sampling for Claude
MAX_FRAME_WIDTH = int(os.environ.get("MAX_FRAME_WIDTH", "768"))
# Region for boto3 service clients
REGION = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
          or "us-east-1")


def segment_key(s3_key: str) -> str:
    """Map an input key to its segment JSON key under segment/."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    return f"{SEGMENT_PREFIX}/{stem}.json"


def _should_run_content_labeling() -> bool:
    """Whether to label content spans with Claude (via Bedrock, no key needed)."""
    return CONTENT_SEGMENT != "off"


# --- Transcription ---
def turns_to_vtt(turns: list) -> str:
    """Convert speaker turns to WebVTT format (for Aviary/captioning)."""
    lines = ["WEBVTT", ""]
    for i, t in enumerate(turns, 1):
        start_h = int(t["start"] // 3600)
        start_m = int((t["start"] % 3600) // 60)
        start_s = t["start"] % 60
        end_h = int(t["end"] // 3600)
        end_m = int((t["end"] % 3600) // 60)
        end_s = t["end"] % 60
        lines.append(str(i))
        lines.append(f"{start_h:02d}:{start_m:02d}:{start_s:06.3f} --> "
                     f"{end_h:02d}:{end_m:02d}:{end_s:06.3f}")
        speaker = t.get("speaker", "")
        text = t.get("text", "")
        if speaker and speaker != "spk_?":
            lines.append(f"<v {speaker}>{text}")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines)


def run_transcription(s3_bucket: str, s3_key: str) -> list:
    """Run AWS Transcribe on the video and return diarized speaker turns.

    Transcribe reads the video directly from S3 (no local copy needed for audio).
    Returns: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    """
    import boto3 as _boto3
    from transcribe import start_job, wait, fetch_result, to_speaker_turns

    stem = Path(s3_key).name.rsplit(".", 1)[0]
    job_name = f"scua-{stem}-{int(time.time())}"[:200]

    transcribe_client = _boto3.client("transcribe", region_name=REGION)
    s3_client = _boto3.client("s3", region_name=REGION)

    media_uri = f"s3://{s3_bucket}/{s3_key}"
    log.info(f"[Transcribe] Starting job {job_name} on {media_uri}")
    start_job(transcribe_client, media_uri, job_name=job_name,
              language=TRANSCRIBE_LANGUAGE, max_speakers=MAX_SPEAKERS)
    job = wait(transcribe_client, job_name)
    turns = to_speaker_turns(fetch_result(job, s3_client))
    speakers = sorted({t["speaker"] for t in turns})
    log.info(f"[Transcribe] {len(turns)} speaker turns, {len(speakers)} speakers")
    return turns


def transcript_for_span(turns: list, start: float, end: float) -> str:
    """Extract transcript text for words that overlap a time span."""
    words = []
    for t in turns:
        # Include turn if it overlaps with the span
        if t["end"] > start and t["start"] < end:
            words.append(t["text"])
    return " ".join(words).strip()


# --- Frame sampling ---
def sample_frame(video_path: str, timestamp: float, max_width: int = 768) -> str:
    """Extract a single frame at `timestamp` and return as base64 JPEG."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return ""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --- Content labeling with Claude ---
def label_content_spans(video_path: str, content_spans, duration: float,
                        turns: list = None) -> list:
    """Label each content span using Claude Sonnet (1 API call per span).

    For each content span, samples a frame from the midpoint and asks Claude
    to describe what's happening — including the transcript for context if
    available. Returns a list of dicts with title, description, and transcript.
    """
    from bedrock import make_client, BEDROCK_MODEL

    client = make_client()
    model = BEDROCK_MODEL
    labeled = []

    for i, cs in enumerate(content_spans):
        mid = (cs.start + cs.end) / 2.0
        span_dur = cs.end - cs.start
        frame_b64 = sample_frame(video_path, mid, MAX_FRAME_WIDTH)

        # Get transcript for this span
        span_transcript = ""
        if turns:
            span_transcript = transcript_for_span(turns, cs.start, cs.end)

        # Build the prompt — include transcript snippet for context
        transcript_context = ""
        if span_transcript:
            # Limit to first 500 chars to control token usage
            snippet = span_transcript[:500]
            if len(span_transcript) > 500:
                snippet += "..."
            transcript_context = f"\n\nTranscript excerpt from this segment:\n\"{snippet}\""

        prompt_text = (
            f"This frame is from an archival video at timestamp {mid:.0f}s "
            f"(content span {cs.start:.0f}s to {cs.end:.0f}s, duration {span_dur:.0f}s, "
            f"video total {duration:.0f}s).{transcript_context}\n\n"
            f"Describe what is happening in this segment in 1-2 sentences. "
            f"Then provide a short title (max 8 words) for this content segment. "
            f"Respond in JSON: {{\"title\": \"...\", \"description\": \"...\"}}"
        )

        content_blocks = []
        if frame_b64:
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}
            })
        content_blocks.append({"type": "text", "text": prompt_text})

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": content_blocks}],
            )
            text = resp.content[0].text.strip()
            # Handle cases where Claude wraps in ```json ... ```
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            title = data.get("title", "Content").strip()
            description = data.get("description", "").strip()
        except Exception as e:
            log.warning(f"Claude labeling failed for span {i} ({cs.start:.1f}-{cs.end:.1f}s): {e}")
            title = "Content"
            description = ""

        labeled.append({
            "start": cs.start,
            "end": cs.end,
            "title": title,
            "description": description,
            "content_label": "content",
            "transcript": span_transcript,
        })
        log.info(f"  Span {i+1}/{len(content_spans)} "
                 f"[{cs.start:.1f}-{cs.end:.1f}s]: {title}")

    return labeled


# --- Build output JSON ---
def build_segment_json(s3_key: str, analysis, labeled_content=None) -> dict:
    """Build alternating D/C/I segment JSON for the SCUA Editor.

    Interleaves dead spans (type "D") with content spans (type "C" if labeled,
    "I" if labeling was skipped/failed) sorted by start time.
    """
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    segments = []

    # Build a unified timeline: dead + content, sorted by start
    entries = []
    for ds in analysis.dead_spans:
        entries.append(("D", ds.start, ds.end, ds.kind, None))

    if labeled_content:
        for lc in labeled_content:
            entries.append(("C", lc["start"], lc["end"], None, lc))
    else:
        # No labeling — mark all content spans as generic "I"
        for cs in analysis.content_spans:
            entries.append(("I", cs.start, cs.end, None, None))

    entries.sort(key=lambda x: x[1])

    for seg_type, start, end, dead_kind, label_data in entries:
        seg = {
            "segment_start": round(start, 2),
            "segment_end": round(end, 2),
            "segment_type": seg_type,
        }
        if seg_type == "D":
            seg["title"] = f"Dead space ({dead_kind})" if dead_kind else "Dead space"
        elif seg_type == "C" and label_data:
            seg["title"] = label_data.get("title", "Content")
            seg["content_label"] = label_data.get("content_label", "content")
            if label_data.get("description"):
                seg["description"] = label_data["description"]
            if label_data.get("transcript"):
                seg["transcript"] = label_data["transcript"]
        else:
            seg["title"] = "Content"

        segments.append(seg)

    total_content = sum(cs.end - cs.start for cs in analysis.content_spans)
    kept_pct = (100 * total_content / analysis.duration) if analysis.duration else 0

    return {
        "video": stem,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auto_detected": True,
        "content_segmented": labeled_content is not None,
        "status": analysis.status,
        "duration": round(analysis.duration, 2),
        "kept_pct": round(kept_pct, 1),
        "dead_span_count": len(analysis.dead_spans),
        "content_span_count": len(analysis.content_spans),
        "segments": segments,
    }


def write_review_marker(s3_client, bucket: str, s3_key: str, analysis) -> None:
    """Drop a review marker so a human can find flagged files."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    total_dead = sum(ds.end - ds.start for ds in analysis.dead_spans)
    total_content = sum(cs.end - cs.start for cs in analysis.content_spans)
    body = (
        f"source: s3://{bucket}/{s3_key}\n"
        f"status: {analysis.status}\n"
        f"duration: {analysis.duration:.1f}s\n"
        f"dead spans: {len(analysis.dead_spans)} ({total_dead:.1f}s)\n"
        f"content spans: {len(analysis.content_spans)} ({total_content:.1f}s)\n"
        + "".join(f"note: {n}\n" for n in analysis.notes)
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
                seg = keep_segments[0]
                log.info(f"Trimming single segment: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                apply_trim(str(local_video_path), str(trimmed_path), seg["start"], seg["end"])
            else:
                import subprocess
                clip_paths = []
                for i, seg in enumerate(keep_segments):
                    clip_path = temp_dir / f"clip_{i:03d}.mp4"
                    log.info(f"  Cutting clip {i}: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                    apply_trim(str(local_video_path), str(clip_path), seg["start"], seg["end"])
                    clip_paths.append(clip_path)

                concat_file = temp_dir / "concat.txt"
                with open(concat_file, "w") as f:
                    for cp in clip_paths:
                        f.write(f"file '{cp}'\n")

                log.info(f"Concatenating {len(clip_paths)} clips...")
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_file), "-c", "copy", str(trimmed_path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, check=True,
                )

            output_key = f"edit/{video_name}_trimmed.mp4"
            log.info(f"Uploading trimmed video to s3://{s3_bucket}/{output_key}")
            s3_client.upload_file(str(trimmed_path), s3_bucket, output_key)

            log.info(f"=== SCUA Video Trimmer Finished in "
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
    """Full-video dead-space detection + content labeling."""
    log.info("=== SCUA Full-Video Segment Detector Starting ===")
    log.info(f"Processing s3://{s3_bucket}/{s3_key}  (mode={TRIM_MODE})")
    s3_client = boto3.client("s3")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            filename = Path(s3_key).name
            local_video_path = temp_dir / filename
            log.info(f"Downloading video to {local_video_path}...")
            s3_client.download_file(s3_bucket, s3_key, str(local_video_path))

            # --- STAGE 1: FULL-VIDEO DEAD-SPACE DETECTION ---
            stage1_start = time.time()
            analysis = analyze_full_video(
                str(local_video_path),
                mode=TRIM_MODE,
                merge_gap=MERGE_GAP,
                min_dead_dur=MIN_DEAD_DUR,
                min_content_dur=MIN_CONTENT_DUR,
                black_pic_th=BLACK_PIC_TH,
            )
            total_dead = sum(ds.end - ds.start for ds in analysis.dead_spans)
            total_content = sum(cs.end - cs.start for cs in analysis.content_spans)
            log.info(
                f"--- Stage 1 (Full-video probe) completed in "
                f"{time.time() - stage1_start:.2f}s ---\n"
                f"  Duration: {analysis.duration:.1f}s | "
                f"Dead spans: {len(analysis.dead_spans)} ({total_dead:.1f}s) | "
                f"Content spans: {len(analysis.content_spans)} ({total_content:.1f}s) | "
                f"Status: {analysis.status}"
            )
            for ds in analysis.dead_spans:
                log.info(f"  DEAD [{ds.start:.1f}s -> {ds.end:.1f}s] "
                         f"{ds.kind} ({ds.end - ds.start:.1f}s)")
            for cs in analysis.content_spans:
                log.info(f"  CONTENT [{cs.start:.1f}s -> {cs.end:.1f}s] "
                         f"({cs.end - cs.start:.1f}s)")
            for n in analysis.notes:
                log.info(f"  note: {n}")

            # --- STAGE 2: TRANSCRIPTION (cached — reuses existing transcript on re-runs) ---
            stage2_start = time.time()
            turns = None
            transcript_s3_key = f"transcript/{Path(s3_key).name.rsplit('.', 1)[0]}.json"

            if analysis.content_spans:
                # Try to load cached transcript from S3 (from a previous run)
                try:
                    resp = s3_client.get_object(Bucket=s3_bucket, Key=transcript_s3_key)
                    turns = json.loads(resp["Body"].read().decode("utf-8"))
                    log.info(f"[Transcribe] Loaded cached transcript from "
                             f"s3://{s3_bucket}/{transcript_s3_key} "
                             f"({len(turns)} turns)")
                except (ClientError, Exception):
                    turns = None

                # Only run Transcribe if no cached transcript exists
                if turns is None:
                    log.info("No cached transcript — running AWS Transcribe...")
                    try:
                        turns = run_transcription(s3_bucket, s3_key)
                        total_speech = sum(t["end"] - t["start"] for t in turns)
                        log.info(f"--- Stage 2 (Transcribe) completed in "
                                 f"{time.time() - stage2_start:.2f}s --- "
                                 f"{len(turns)} turns, {total_speech:.0f}s of speech")
                        # Cache the transcript to S3 for future re-runs
                        s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=transcript_s3_key,
                            Body=json.dumps(turns, indent=2).encode("utf-8"),
                            ContentType="application/json",
                        )
                        # Also write VTT for Aviary / captioning
                        vtt_key = transcript_s3_key.replace(".json", ".vtt")
                        s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=vtt_key,
                            Body=turns_to_vtt(turns).encode("utf-8"),
                            ContentType="text/vtt",
                        )
                        log.info(f"  Cached transcript to s3://{s3_bucket}/{transcript_s3_key}")
                        log.info(f"  Wrote VTT to s3://{s3_bucket}/{vtt_key}")
                    except Exception as e:
                        log.warning(f"Transcription failed (continuing without transcript): {e}")
                        log.warning(traceback.format_exc())
                        turns = None
                else:
                    log.info(f"--- Stage 2 (Transcribe) skipped — using cached transcript "
                             f"({time.time() - stage2_start:.2f}s) ---")
            else:
                log.info("No content spans — skipping transcription")

            # --- STAGE 3: CONTENT LABELING (Claude Sonnet) ---
            stage3_start = time.time()
            labeled_content = None

            if (_should_run_content_labeling()
                    and analysis.status == "OK"
                    and analysis.content_spans):
                log.info(f"Labeling {len(analysis.content_spans)} content spans "
                         f"with Claude Sonnet...")
                try:
                    labeled_content = label_content_spans(
                        str(local_video_path),
                        analysis.content_spans,
                        analysis.duration,
                        turns=turns,
                    )
                    log.info(f"Content labeling complete in "
                             f"{time.time() - stage3_start:.2f}s")
                except Exception as e:
                    log.warning(f"Content labeling failed (falling back to unlabeled): {e}")
                    log.warning(traceback.format_exc())
                    labeled_content = None
            elif not _should_run_content_labeling():
                log.info("Content labeling disabled (no ANTHROPIC_API_KEY or CONTENT_SEGMENT=off)")
            else:
                log.info("Skipping content labeling (status not OK or no content spans)")

            # If labeling was skipped but we have transcripts, still attach them
            if labeled_content is None and turns and analysis.content_spans:
                labeled_content = []
                for cs in analysis.content_spans:
                    labeled_content.append({
                        "start": cs.start,
                        "end": cs.end,
                        "title": "Content",
                        "description": "",
                        "content_label": "content",
                        "transcript": transcript_for_span(turns, cs.start, cs.end),
                    })

            # --- STAGE 4: WRITE SEGMENTS ---
            stage4_start = time.time()
            seg_s3_key = segment_key(s3_key)

            if analysis.status == "NEEDS_REVIEW":
                write_review_marker(s3_client, s3_bucket, s3_key, analysis)

            seg_json = build_segment_json(s3_key, analysis, labeled_content)
            log.info(f"Uploading segment JSON to s3://{s3_bucket}/{seg_s3_key} "
                     f"({len(seg_json['segments'])} segments)")
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=seg_s3_key,
                Body=json.dumps(seg_json, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            log.info(f"--- Stage 4 (Write) completed in "
                     f"{time.time() - stage4_start:.2f}s ---")

        except ClientError as e:
            log.error(f"S3 error: {e.response['Error']['Message']}")
            log.error(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.error(traceback.format_exc())
            sys.exit(1)

    log.info(f"=== SCUA Segment Detector Finished in "
             f"{time.time() - pipeline_start_time:.2f}s ===")


if __name__ == "__main__":
    main()
