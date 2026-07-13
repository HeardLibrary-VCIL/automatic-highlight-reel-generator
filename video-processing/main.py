"""ECS entry point: download video from S3, detect dead space, write segments.

Integrated with SCUA-Video-Editing Amplify frontend:
  - Input  : S3_BUCKET / S3_KEY pointing at uploaded video under `video/` prefix
  - Output : segment JSON  → `segment/{basename}.json`  (for the Editor timeline)
             review marker → `review/{basename}.txt`    (if NEEDS_REVIEW)

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


def segment_key(s3_key: str) -> str:
    """Map an input key to its segment JSON key under segment/."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    return f"{SEGMENT_PREFIX}/{stem}.json"


def build_segment_json(s3_key: str, prop) -> dict:
    """Build a segment JSON matching SCUA Editor format from the trim proposal.

    Creates segments:
      - D (Deadspace) for detected head/tail dead regions
      - I (Interview/Content) for the kept content span
    """
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    segments = []

    # Head deadspace
    if prop.content_start > 1.0:
        segments.append({
            "segment_start": 0,
            "segment_end": round(prop.content_start, 2),
            "segment_type": "D",
            "title": "Head dead space (auto-detected)",
        })

    # Main content
    segments.append({
        "segment_start": round(prop.content_start, 2),
        "segment_end": round(prop.content_end, 2),
        "segment_type": "I",
        "title": "Content",
    })

    # Tail deadspace
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

            # --- STAGE 2: WRITE SEGMENTS ---
            stage2_start = time.time()
            seg_s3_key = segment_key(s3_key)

            if prop.status == "NEEDS_REVIEW":
                write_review_marker(s3_client, s3_bucket, s3_key, prop)

            # Always write segment JSON so the Editor can display it
            seg_json = build_segment_json(s3_key, prop)
            log.info(f"Uploading segment JSON to s3://{s3_bucket}/{seg_s3_key}")
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=seg_s3_key,
                Body=json.dumps(seg_json, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            log.info(f"--- Stage 2 (Write Segments) completed in "
                     f"{time.time() - stage2_start:.2f}s ---")

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
