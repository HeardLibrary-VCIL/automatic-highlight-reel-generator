# Slim CPU image for dead-space trimming + content-type segmentation, and the
# audio-only pipeline in ../audio-processing.
# Uses ffmpeg (black/freeze/silence, audio decoding) + opencv (bars/snow + frame
# sampling) + scenedetect (shot boundaries) + anthropic (Claude via Bedrock) +
# boto3 (S3 I/O) + numpy/pydantic (audio segmentation + label validation).
# CPU-only: content classification runs on Amazon Bedrock (Claude), not locally.
#
# BUILD CONTEXT = THE REPO ROOT (so the image can include both folders):
#   docker build -f video-processing/Dockerfile .
# CDK does the same (lib/highlight-processor-stack.ts: fromAsset('.', {file: ...})).
# What gets sent to the build is limited by video-processing/Dockerfile.dockerignore
# locally and by the matching `exclude` list in the CDK stack.
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg powers the detectors, the trim and audio decoding. libglib2.0-0 is a
# runtime dep of opencv-python-headless. Kept intentionally minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY video-processing/requirements-trim.txt ./requirements-trim.txt
COPY audio-processing/requirements.txt ./requirements-audio.txt
RUN pip3 install --no-cache-dir -r requirements-trim.txt -r requirements-audio.txt

# Application code: the video pipeline at /app (main.py is the entry point) and
# the audio pipeline at /app/audio-processing (main.py adds it to sys.path).
COPY video-processing/ /app/
COPY audio-processing/ /app/audio-processing/

# The ECS task overrides env (S3_BUCKET, S3_KEY) and runs this.
CMD ["python3", "main.py"]
