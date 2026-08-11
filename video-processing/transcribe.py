"""Stage A of the audio-fusion pipeline: transcribe an S3 video with Amazon
Transcribe and emit a shot-joinable `transcript.json`.

Why Amazon Transcribe (not local Whisper): the archive already lives in S3, the
service reads mp4 straight from S3 (no local audio extraction), and speaker
diarization ("speaker labels") is a built-in flag -- exactly the 1-speaker
(host/PSA monologue) vs many-speaker (interview) signal we need, with no local
GPU or pyannote/HuggingFace setup. NOTE: Anthropic/Claude does not do speech-to-
text, so ASR has to come from a dedicated service; this is that step.

Batch Transcribe only reads from S3 (Media.MediaFileUri must be an s3:// URI) and
writes its result to S3 (your --output-bucket, or a service-managed bucket that
hands back a presigned URL). This script starts the job, polls to completion,
downloads the raw result, and flattens it into speaker turns:

    transcript.json = [{"start": <s>, "end": <s>, "speaker": "spk_0",
                        "text": "..."}, ...]

That is the unit Stage B joins onto each shot from segment_shots.py.

Usage:
    python transcribe.py s3://bucket/video/RCC_183.mp4 -o transcript.json
    python transcribe.py s3://.../RCC_183.mp4 --profile scua --region us-east-1
    python transcribe.py s3://.../RCC_183.mp4 --output-bucket my-bucket --output-prefix transcribe/
    python transcribe.py --raw asrOutput.json -o transcript.json   # just re-parse a saved result

Needs boto3 (already in requirements-trim.txt) and AWS creds with
transcribe:StartTranscriptionJob + s3:GetObject on the media (and s3:PutObject on
--output-bucket if you set one).
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone


def _media_format(uri: str) -> str:
    ext = os.path.splitext(uri)[1].lstrip(".").lower()
    return {"mov": "mp4", "m4a": "mp4", "qt": "mp4"}.get(ext, ext or "mp4")


def start_job(client, media_uri, *, job_name, language="en-US",
              max_speakers=10, output_bucket=None, output_prefix=""):
    """Kick off one batch job with speaker diarization on. Returns job_name."""
    settings = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": max_speakers}
    kwargs = dict(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat=_media_format(media_uri),
        LanguageCode=language,
        Settings=settings,
    )
    if output_bucket:
        kwargs["OutputBucketName"] = output_bucket
        if output_prefix:
            kwargs["OutputKey"] = output_prefix.rstrip("/") + "/" + job_name + ".json"
    client.start_transcription_job(**kwargs)
    return job_name


def wait(client, job_name, poll=10.0):
    """Block until the job finishes; return the job dict (raises on FAILED)."""
    while True:
        job = client.get_transcription_job(TranscriptionJobName=job_name)["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        print(f"  [{datetime.now():%H:%M:%S}] {job_name}: {status}", file=sys.stderr)
        if status == "COMPLETED":
            return job
        if status == "FAILED":
            raise RuntimeError(f"Transcribe job failed: {job.get('FailureReason')}")
        time.sleep(poll)


def fetch_result(job, s3_client) -> dict:
    """Download the raw Transcribe result JSON from wherever the job wrote it."""
    uri = job["Transcript"]["TranscriptFileUri"]
    if uri.startswith("s3://"):                 # --output-bucket path: read via S3
        _, _, rest = uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with urllib.request.urlopen(uri) as r:      # service-managed path: presigned HTTPS URL
        return json.loads(r.read())


# ------------------------------- flatten to turns -----------------------------
def to_speaker_turns(raw: dict) -> list:
    """Collapse Transcribe's word-level output into speaker turns.

    Transcribe gives results.items[] (each pronunciation item has start_time/
    end_time/alternatives) and results.speaker_labels.segments[] (time ranges
    tagged spk_N). We map each timed word to its content, then walk the speaker
    segments and gather the words that fall inside each -> one row per turn.
    Falls back to sentence-ish grouping if diarization was off.
    """
    results = raw.get("results", raw)
    items = results.get("items", [])
    # start_time -> spoken word, for pronunciation items only (punctuation has no time)
    word_at = {}
    for it in items:
        if it.get("type") == "pronunciation" and it.get("start_time"):
            alts = it.get("alternatives") or [{}]
            word_at[it["start_time"]] = alts[0].get("content", "")

    segments = results.get("speaker_labels", {}).get("segments", [])
    turns = []
    if segments:
        for seg in segments:
            words = [word_at.get(w.get("start_time"), "")
                     for w in seg.get("items", []) if w.get("start_time") in word_at]
            text = " ".join(w for w in words if w).strip()
            if not text:
                continue
            turns.append({
                "start": round(float(seg["start_time"]), 2),
                "end": round(float(seg["end_time"]), 2),
                "speaker": seg.get("speaker_label", "spk_?"),
                "text": text,
            })
        return _merge_same_speaker(turns)

    # No diarization: emit the flat transcript as a single turn (better than nothing).
    text = (results.get("transcripts") or [{}])[0].get("transcript", "").strip()
    if text:
        starts = [float(it["start_time"]) for it in items if it.get("start_time")]
        ends = [float(it["end_time"]) for it in items if it.get("end_time")]
        turns.append({"start": round(min(starts), 2) if starts else 0.0,
                      "end": round(max(ends), 2) if ends else 0.0,
                      "speaker": "spk_?", "text": text})
    return turns


def _merge_same_speaker(turns: list) -> list:
    """Join back-to-back turns by the same speaker (Transcribe splits on pauses)."""
    out = []
    for t in turns:
        if out and out[-1]["speaker"] == t["speaker"] and t["start"] - out[-1]["end"] < 3.0:
            out[-1]["end"] = t["end"]
            out[-1]["text"] += " " + t["text"]
        else:
            out.append(dict(t))
    return out


# ------------------------------------ main ------------------------------------
def main():
    p = argparse.ArgumentParser(description="Transcribe an S3 video with Amazon Transcribe.")
    p.add_argument("media_uri", nargs="?", help="s3://bucket/key of the video (mp4).")
    p.add_argument("-o", "--output", default="transcript.json")
    p.add_argument("--raw", help="Skip AWS; re-parse a saved raw Transcribe JSON into transcript.json.")
    p.add_argument("--profile", default=None)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--language", default="en-US")
    p.add_argument("--max-speakers", type=int, default=10)
    p.add_argument("--job-name", default=None, help="Defaults to <basename>-<timestamp>.")
    p.add_argument("--output-bucket", default=None,
                   help="Write the raw result to this bucket (else service-managed + presigned URL).")
    p.add_argument("--output-prefix", default="")
    p.add_argument("--save-raw", default=None, help="Also save the raw Transcribe JSON here.")
    args = p.parse_args()

    if args.raw:
        raw = json.load(open(args.raw))
    else:
        if not args.media_uri:
            p.error("media_uri is required unless --raw is given")
        import boto3
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        transcribe = session.client("transcribe")
        s3 = session.client("s3")

        base = os.path.splitext(os.path.basename(args.media_uri))[0]
        job_name = args.job_name or f"{base}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        print(f"Starting job {job_name} on {args.media_uri}", file=sys.stderr)
        start_job(transcribe, args.media_uri, job_name=job_name, language=args.language,
                  max_speakers=args.max_speakers, output_bucket=args.output_bucket,
                  output_prefix=args.output_prefix)
        job = wait(transcribe, job_name)
        raw = fetch_result(job, s3)
        if args.save_raw:
            with open(args.save_raw, "w") as f:
                json.dump(raw, f)
            print(f"Saved raw result to {args.save_raw}", file=sys.stderr)

    turns = to_speaker_turns(raw)
    with open(args.output, "w") as f:
        json.dump(turns, f, indent=2)

    speakers = sorted({t["speaker"] for t in turns})
    total = sum(t["end"] - t["start"] for t in turns)
    print(f"\n{len(turns)} speaker turns, {len(speakers)} speakers ({', '.join(speakers)}), "
          f"{total:.0f}s of speech -> {args.output}")
    for t in turns[:8]:
        print(f"  [{t['start']:7.1f}->{t['end']:7.1f}] {t['speaker']}: {t['text'][:70]}")


if __name__ == "__main__":
    main()
