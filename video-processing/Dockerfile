# Slim CPU image for dead-space trimming + content-type segmentation.
# Uses ffmpeg (black/freeze/silence) + opencv (bars/snow + frame sampling)
# + scenedetect (shot boundaries) + anthropic (Claude vision classify) + boto3 (S3 I/O).
# CPU-only: content classification runs on Amazon Bedrock (Claude), not locally.
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg powers the detectors + the trim. libglib2.0-0 is a runtime dep of
# opencv-python-headless. Kept intentionally minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-trim.txt .
RUN pip3 install --no-cache-dir -r requirements-trim.txt

# Copy application code (see .dockerignore for what's excluded).
COPY . /app/

# The ECS task overrides env (S3_BUCKET, S3_KEY) and runs this.
CMD ["python3", "main.py"]
