"""Entry point for audio-only files: the one function video-processing/main.py calls.

main.py downloads the upload, asks `is_audio_only(path)`, and if so hands the
file here instead of running the video stages:

    1. Transcribe (main.py's cached helper, passed in) -> what is said / sung, and when.
    2. audio_segment: DSP pass -> songs, speech, applause, silence/hiss/tone/static.
       Transcribe's speech spans are a hint to the music-vs-speech decision.
    3. audio_label: Claude on Bedrock names each song from the announcements around
       it + its lyrics; then a title + summary for the recording.
    4. Write segment/{stem}.json (same shape the Editor already reads for video).

Steps 1 and 3 are best-effort: without them the timeline is still written, with
plain titles ("Song 3"). Dependencies on the video pipeline (Transcribe caching,
the Bedrock client) are passed in as arguments, so this folder imports nothing
from video-processing and can be run and tested on its own.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from audio_segment import is_audio_only, segment_audio  # noqa: F401  (is_audio_only re-exported for main.py)
from audio_label import label_audio, build_audio_segment_json

log = logging.getLogger(__name__)


def run_audio_detect(local_path, s3_client, s3_bucket, s3_key, original_filename, *,
                     get_turns=None, make_client=None, seg_key=None):
    """Segment an audio-only file and upload its segment JSON. Returns the JSON dict.

    get_turns(local_path, bucket, key) -> transcript turns  (None = skip Transcribe)
    make_client() -> a Bedrock (AnthropicBedrock) client     (None = skip labeling)
    seg_key: S3 key for the output (default segment/{stem}.json)
    """
    log.info("=== Audio-only file: running audio segmentation ===")
    turns = []
    if get_turns is not None:
        try:
            turns = get_turns(local_path, s3_bucket, s3_key) or []
            log.info(f"--- Transcribe: {len(turns)} turns ---")
        except Exception as e:
            log.warning(f"Transcription failed ({e}); segmenting from audio alone")

    t0 = time.time()
    regions, rows, _, scores = segment_audio(str(local_path), turns)
    log.info(f"--- Audio segmentation: {len(regions)} regions in {time.time() - t0:.1f}s "
             f"(duration {rows.duration:.1f}s, quiet < {scores['quiet_th']:.1f} dBFS) ---")
    for r in regions:
        log.info(f"  [{r.start:8.2f}s -> {r.end:8.2f}s] {r.kind:9s} "
                 f"cues={','.join(r.cues) or '-'} {r.confidence}")

    labels = {}
    if make_client is not None:
        try:
            labels = label_audio(regions, turns, make_client(), original_filename, rows.duration)
        except Exception as e:
            log.warning(f"Audio labeling skipped (plain titles used): {e}")

    seg_json = build_audio_segment_json(s3_key, regions, turns, labels, rows.duration)
    if seg_key is None:
        seg_key = f"segment/{Path(s3_key).name.rsplit('.', 1)[0]}.json"
    log.info(f"Uploading segment JSON ({seg_json['song_count']} songs) to s3://{s3_bucket}/{seg_key}")
    s3_client.put_object(
        Bucket=s3_bucket, Key=seg_key,
        Body=json.dumps(seg_json, indent=2).encode("utf-8"),
        ContentType="application/json",
        Metadata={"original-filename": original_filename},
    )
    return seg_json
