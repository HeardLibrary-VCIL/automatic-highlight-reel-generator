"""ECS entry point: download a video from S3, trim head/tail dead space, upload the result.

This replaces the old diving-detection pipeline (downsample -> PaliGemma VLM
inference -> clip/merge). The trimming logic lives in analyze_deadspace.py; this
module only handles S3 I/O and the ECS/CloudWatch-facing orchestration.

Contract with the rest of the system (unchanged wiring):
  - Input  : the Lambda passes S3_BUCKET / S3_KEY env vars pointing at the
             uploaded video under the `videos/` prefix.
  - Output : the trimmed video is written under RESULT_PREFIX (default `results/`)
             as `<basename>_trimmed.mp4`, which is what the frontend polls for.
  - Files the trimmer is unsure about (status NEEDS_REVIEW) are handled per
             REVIEW_POLICY and get a marker under `review/`.
"""
import os
import sys
import logging
import tempfile
import traceback
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Import the trimming API (see analyze_deadspace.py)
from analyze_deadspace import analyze, apply_trim

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Where trimmed results land in the bucket (must match the frontend's RESULT_PREFIX).
RESULT_PREFIX = os.environ.get("RESULT_PREFIX", "results").strip("/")
# Trim mode: "black" (conservative, default) or "static" (aggressive).
TRIM_MODE = os.environ.get("TRIM_MODE", "black")
# What to do with files flagged NEEDS_REVIEW:
#   "upload_original" (default) -> upload the untrimmed original so the user still
#                                  gets something back, and drop a review marker.
#   "skip"                      -> upload nothing, only drop a review marker.
REVIEW_POLICY = os.environ.get("REVIEW_POLICY", "upload_original")


def result_key(s3_key: str) -> str:
    """Map an input key to its trimmed-output key under RESULT_PREFIX."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    return f"{RESULT_PREFIX}/{stem}_trimmed.mp4"


def write_review_marker(s3_client, bucket: str, s3_key: str, prop) -> None:
    """Drop a small text marker under review/ so a human can find flagged files."""
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
    except Exception as e:  # a marker failure must not fail the whole job
        log.warning(f"Could not write review marker: {e}")


# --- Main Orchestrator ---
def main():
    pipeline_start_time = time.time()
    log.info("=== Dead-space Trimmer Starting ===")

    # 1. Configuration from environment (set by the trigger Lambda)
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_key = os.environ.get("S3_KEY")
    if not s3_bucket or not s3_key:
        log.error("S3_BUCKET and S3_KEY environment variables are required.")
        sys.exit(1)

    log.info(f"Processing s3://{s3_bucket}/{s3_key}  (mode={TRIM_MODE})")
    s3_client = boto3.client("s3")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        log.info(f"Created temporary working directory: {temp_dir}")

        try:
            filename = Path(s3_key).name
            stem = filename.rsplit(".", 1)[0]
            local_video_path = temp_dir / filename
            log.info(f"Downloading video to {local_video_path}...")
            s3_client.download_file(s3_bucket, s3_key, str(local_video_path))

            # --- STAGE 1: PROBE (detect head/tail dead space) ---
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

            # --- STAGE 2: DECIDE + TRIM ---
            stage2_start = time.time()
            output_s3_key = result_key(s3_key)

            if prop.status == "OK" and prop.kept > 0:
                trimmed_path = temp_dir / f"{stem}_trimmed.mp4"
                log.info(f"Trimming head={prop.head_trim:.2f}s tail={prop.tail_trim:.2f}s ...")
                apply_trim(str(local_video_path), str(trimmed_path),
                           prop.content_start, prop.content_end)
                upload_path, disposition = trimmed_path, "trimmed"
            else:
                # NEEDS_REVIEW or empty span -> do NOT cut (never risk the program).
                write_review_marker(s3_client, s3_bucket, s3_key, prop)
                if REVIEW_POLICY == "skip":
                    log.warning(
                        f"status={prop.status}, REVIEW_POLICY=skip -> not uploading a result; "
                        f"flagged under review/{stem}.txt"
                    )
                    log.info(f"=== Dead-space Trimmer Finished Successfully "
                             f"(flagged NEEDS_REVIEW) in {time.time() - pipeline_start_time:.2f}s ===")
                    return
                log.warning(
                    f"status={prop.status} -> uploading the ORIGINAL untrimmed video and "
                    f"flagging under review/{stem}.txt"
                )
                upload_path, disposition = local_video_path, "original-needs-review"

            log.info(f"--- Stage 2 (Trim/Decide) completed in "
                     f"{time.time() - stage2_start:.2f}s --- ({disposition})")

            # 3. Upload the result
            log.info(f"Uploading result to s3://{s3_bucket}/{output_s3_key} ({disposition})")
            s3_client.upload_file(
                str(upload_path), s3_bucket, output_s3_key,
                ExtraArgs={"Metadata": {
                    "disposition": disposition,
                    "status": prop.status,
                    "kept_pct": f"{prop.kept_pct:.0f}",
                    "head_trim_s": f"{prop.head_trim:.2f}",
                    "tail_trim_s": f"{prop.tail_trim:.2f}",
                }},
            )

        except ClientError as e:
            log.error(f"An S3 error occurred: {e.response['Error']['Message']}")
            log.error(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}")
            log.error(traceback.format_exc())
            sys.exit(1)

    log.info(f"=== Dead-space Trimmer Finished Successfully in "
             f"{time.time() - pipeline_start_time:.2f}s ===")


if __name__ == "__main__":
    main()
