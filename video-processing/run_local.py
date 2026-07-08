"""Run the highlight pipeline fully locally — no S3, no AWS.

This is a drop-in alternative to main.py for running the three pipeline stages
(downsample -> inference -> clip/merge) against a video file on disk. It is
intended for local development/testing (e.g. on an Apple Silicon Mac via MPS).

Usage:
    # From inside the video-processing/ directory (config.yaml is loaded from CWD):
    python run_local.py path/to/video.mp4
    python run_local.py path/to/video.mp4 -o output --target-fps 4 \
        -p "<image> Is there a person diving? Answer with 'yes' or 'no'.\n"

Requirements: ffmpeg + ffprobe on PATH, deps from requirements-local.txt, and a
Hugging Face login with the PaliGemma-2 license accepted (huggingface-cli login).
"""

import os

# Let any MPS op that isn't implemented fall back to CPU instead of crashing.
# Must be set before torch is imported (transitively via the stage modules).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import logging
import sys
import time
from pathlib import Path

from downsample_videos import run_downsampling
from run_inference_and_postprocess import run_inference
from clipping_and_merging import run_clipping
from config_loader import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the highlight pipeline locally.")
    parser.add_argument("input_video", help="Path to the local input video file.")
    parser.add_argument(
        "-o", "--output-dir", default="output",
        help="Directory for intermediate files and the final highlight reel (default: ./output).",
    )
    parser.add_argument(
        "-p", "--prompt", default=None,
        help="Detection prompt. Defaults to main.default_prompt from config.yaml.",
    )
    parser.add_argument(
        "--target-fps", type=int, default=config["downsampling"]["target_fps"],
        help="FPS to downsample to before inference (default from config.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start = time.time()

    input_path = Path(args.input_video).expanduser().resolve()
    if not input_path.exists():
        log.error(f"Input video not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt or config["main"]["default_prompt"]

    log.info("=== Local Video Highlight Processor Starting ===")
    log.info(f"Input:  {input_path}")
    log.info(f"Output: {output_dir}")
    log.info(f"Prompt: {prompt!r}")

    # --- STAGE 1: DOWNSAMPLING ---
    stage_start = time.time()
    downsampled_video_path, timestamps_csv_path = run_downsampling(
        input_video_path=input_path,
        output_dir=output_dir,
        target_fps=args.target_fps,
    )
    log.info(f"--- Stage 1 (Downsampling) completed in {time.time() - stage_start:.2f}s ---")

    # --- STAGE 2: INFERENCE & POST-PROCESSING ---
    stage_start = time.time()
    predicted_intervals_csv_path = run_inference(
        downsampled_video_path=downsampled_video_path,
        timestamps_csv_path=timestamps_csv_path,
        output_dir=output_dir,
        prompt=prompt,
        inference_config=config["inference"],
        post_proc_config=config["post_processing"],
        target_fps=args.target_fps,
    )
    log.info(f"--- Stage 2 (Inference) completed in {time.time() - stage_start:.2f}s ---")

    # --- STAGE 3: CLIPPING & MERGING ---
    stage_start = time.time()
    final_video_path = run_clipping(
        original_video_path=input_path,
        predicted_intervals_csv_path=predicted_intervals_csv_path,
        output_dir=output_dir,
        clipping_config=config["clipping"],
    )
    log.info(f"--- Stage 3 (Clipping & Merging) completed in {time.time() - stage_start:.2f}s ---")

    if final_video_path and final_video_path.exists():
        log.info(f"✅ Highlight reel written to: {final_video_path}")
    else:
        log.warning("No highlights were produced (no intervals detected for this prompt/threshold).")

    log.info(f"=== Finished in {time.time() - pipeline_start:.2f}s ===")


if __name__ == "__main__":
    main()
