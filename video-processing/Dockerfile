# Slim CPU image for head/tail dead-space trimming.
# Uses ffmpeg (black/freeze/silence) + opencv (bars/snow) + boto3 (S3 I/O).
# No GPU, PyTorch, transformers, or PaliGemma -- that was the old diving pipeline.
# (Recover the old GPU Dockerfile from git history if the VLM path is needed again.)
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
