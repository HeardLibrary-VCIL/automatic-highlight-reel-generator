"""ECS entry point: download video from S3, detect dead space, segment content, write results.

Integrated with SCUA-Video-Editing Amplify frontend:
  - Input  : S3_BUCKET / S3_KEY pointing at uploaded video under `video/` prefix
  - Output : segment JSON  → `segment/{basename}.json`  (for the Editor timeline)
             review marker → `review/{basename}.txt`    (if NEEDS_REVIEW)

Pipeline stages:
  1. Dead-space detection (analyze_deadspace): find head/tail dead regions
  2. Content-type segmentation (segment_shots): classify content spans into
     a closed category set (dance, football, tv show, interview, political ad,
     PSA, or 'other') using shot-boundary detection + Claude vision API
  3. Write combined segment JSON to S3 for the Editor UI

The actual video trimming is triggered later by the user from the Editor UI,
after they review and adjust the auto-detected segments.
"""
import json
import os
import sys
import logging
import tempfile
import traceback
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from analyze_deadspace import analyze, apply_trim
from segment_shots import segment_video_shots

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
# Content-type segmentation: enabled by default when ANTHROPIC_API_KEY is set
CONTENT_SEGMENT = os.environ.get("CONTENT_SEGMENT", "auto")  # "auto", "on", "off"
SEGMENT_DETECTOR = os.environ.get("SEGMENT_DETECTOR", "adaptive")  # "adaptive" or "content"
SEGMENT_MIN_SHOT = float(os.environ.get("SEGMENT_MIN_SHOT", "1.5"))


def segment_key(s3_key: str) -> str:
    """Map an input key to its segment JSON key under segment/."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    return f"{SEGMENT_PREFIX}/{stem}.json"


def _should_run_content_segmentation() -> bool:
    """Decide whether to run content-type segmentation (needs ANTHROPIC_API_KEY)."""
    if CONTENT_SEGMENT == "off":
        return False
    if CONTENT_SEGMENT == "on":
        return True
    # "auto": run only if the API key is available
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def build_segment_json(s3_key: str, prop, content_segments=None) -> dict:
    """Build a segment JSON matching SCUA Editor format.

    If `content_segments` is provided (from segment_shots), use those richer
    labels (dance performance, football game, etc.). Otherwise fall back to the
    simple D/I dead-space segments.
    """
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    segments = []

    if content_segments:
        # Use the shot-based content-type segmentation results
        # Still include head/tail dead-space markers if present
        if prop.content_start > 1.0:
            segments.append({
                "segment_start": 0,
                "segment_end": round(prop.content_start, 2),
                "segment_type": "D",
                "title": "Head dead space (auto-detected)",
            })

        for seg in content_segments:
            # Map content-type labels to segment_type codes for the Editor:
            #   Target categories → "C" (content, labeled)
            #   "other" → "I" (generic content / interview)
            seg_type = "I" if seg.label == "other" else "C"
            entry = {
                "segment_start": round(seg.start, 2),
                "segment_end": round(seg.end, 2),
                "segment_type": seg_type,
                "title": seg.label.title(),
                "content_label": seg.label,
            }
            if seg.note:
                entry["note"] = seg.note
            segments.append(entry)

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

    return {
        "video": stem,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auto_detected": True,
        "content_segmented": content_segments is not None,
        "trim_status": prop.status,
        "duration": round(prop.duration, 2),
        "kept_pct": round(prop.kept_pct, 1),
        "segments": segments,
    }


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
                import subprocess
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

                # Concatenate
                log.info(f"Concatenating {len(clip_paths)} clips...")
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_file), "-c", "copy", str(trimmed_path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, check=True,
                )

            # Upload trimmed video
            output_key = f"edit/{video_name}_trimmed.mp4"
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

            # --- STAGE 1: PROBE ---
            stage1_start = time.time()
            prop = analyze(str(local_video_path), mode=TRIM_MODE)
            log.info(
                f"--- Stage 1 (Probe) completed in {time.time() - stage1_start:.2f}s --- "
                f"content {prop.content_start:.2f}s -> {prop.content_end:.2f}s | "
                f"head {prop.head_trim:.2f}s, tail {prop.tail_trim:.2f}s | "
                f"kept {prop.kept_pct:.0f}% | status={prop.status}"
            )
            for n in prop.notes:
                log.info(f"  note: {n}")

            # --- STAGE 2: CONTENT SEGMENTATION ---
            stage2_start = time.time()
            content_segments = None

            if _should_run_content_segmentation() and prop.status == "OK" and prop.kept > 0:
                log.info("Running content-type segmentation (segment_shots)...")
                try:
                    _, _, content_segments = segment_video_shots(
                        str(local_video_path),
                        detector=SEGMENT_DETECTOR,
                        min_shot=SEGMENT_MIN_SHOT,
                        trim=True,         # use dead-space window
                        classify=True,
                        smooth=True,
                    )
                    log.info(
                        f"Content segmentation complete: {len(content_segments)} segments "
                        f"in {time.time() - stage2_start:.2f}s"
                    )
                    for s in content_segments:
                        label = f"{s.label} ({s.note})" if s.note else s.label
                        log.info(f"  [{s.start:.2f}s -> {s.end:.2f}s] {label}")
                except Exception as e:
                    log.warning(f"Content segmentation failed (falling back to dead-space only): {e}")
                    log.warning(traceback.format_exc())
                    content_segments = None
            elif not _should_run_content_segmentation():
                log.info("Content segmentation disabled (no ANTHROPIC_API_KEY or CONTENT_SEGMENT=off)")
            else:
                log.info("Skipping content segmentation (trim status not OK or no content)")

            # --- STAGE 3: WRITE SEGMENTS ---
            stage3_start = time.time()
            seg_s3_key = segment_key(s3_key)

            if prop.status == "NEEDS_REVIEW":
                write_review_marker(s3_client, s3_bucket, s3_key, prop)

            # Always write segment JSON so the Editor can display it
            seg_json = build_segment_json(s3_key, prop, content_segments)
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
