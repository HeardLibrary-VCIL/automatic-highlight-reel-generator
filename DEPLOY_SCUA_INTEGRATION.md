# SCUA Video Segment Detector — Deployment Guide

## Overview

This deploys the video segment detector as a backend processing pipeline for the SCUA-Video-Editing Amplify app. When a video is uploaded through the frontend to `video/`, it automatically:

1. Detects dead space throughout the entire video (black, white, bars, snow, freeze+silence)
2. Transcribes audio with speaker diarization (AWS Transcribe, cached for re-runs)
3. Labels each content span using Claude Sonnet (frame + transcript context)
4. Writes segment JSON + transcript JSON + VTT to S3
5. The SCUA Editor loads these for archivist review/editing

## Architecture

```
SCUA Frontend (Amplify)
    │ upload to video/*
    ▼
S3 Bucket (Amplify-managed: scua-video-storage)
    │ S3 event notification
    ▼
Lambda (validate file + launch ECS task)
    │
    ▼
ECS Task (c5.xlarge, CPU-only, ASG min 0 / max 2)
    ├─ Stage 1: Full-video dead-space detection (ffmpeg + OpenCV)
    ├─ Stage 2: Transcription (AWS Transcribe → cached to transcript/{id}.json + .vtt)
    ├─ Stage 3: Content labeling (Claude Sonnet 4.6 via Amazon Bedrock)
    └─ Stage 4: Write segment/{id}.json
    │
    ▼
SCUA Editor reads segment/{id}.json + transcript/{id}.json
    │ archivist reviews, edits boundaries/transcripts, approves
    ▼
Trim request → Lambda → ECS Task (MODE=trim) → edit/{id}_trimmed.mp4
```

## Prerequisites

- AWS CDK installed (`npm install -g aws-cdk`)
- Docker running locally
- AWS CLI configured with a profile (e.g., `scua-video`)
- The SCUA-Video-Editing Amplify app deployed (you need the bucket name)

## Step 1: Get the Amplify Bucket Name

From the SCUA-Video-Editing project:
```bash
cat ../SCUA-Video-Editing/amplify_outputs.json | jq -r '.storage.bucket_name'
```

Returns something like: `amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1`

## Step 2: Deploy CDK Dependencies

```bash
cd Project2/automatic-highlight-reel-generator
npm install
```

## Step 3: Bootstrap CDK (first time only)

```bash
cdk bootstrap aws://ACCOUNT_ID/us-east-1 --profile scua-video
```

## Step 4: Deploy

```bash
cdk deploy \
  --parameters AmplifyBucketName=amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1 \
  --profile scua-video
```

This creates:
- VPC with 2 AZs + NAT gateway
- ECS Cluster with c5.xlarge ASG (scales to 0 when idle)
- Docker image built from `video-processing/` (pushed to ECR)
- Lambda trigger for S3 `video/*` uploads and `edit/*_trim_request.json`
- S3 notification config on the Amplify bucket
- IAM roles (task: S3 + Transcribe + Bedrock; execution: ECR)
- CloudWatch log group: `/ecs/scua-video-processor`
- Bucket lifecycle rule (noncurrent version cleanup)

## Step 6: Test

Upload a video through the SCUA frontend or directly to S3:

```bash
# Copy from existing archive bucket
aws s3 cp \
  "s3://scua-video/10/Web Copy/RCC_10.mp4" \
  "s3://amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1/video/RCC_10.mp4" \
  --profile scua-video
```

Monitor processing in CloudWatch:
```bash
aws logs tail /ecs/scua-video-processor --follow --profile scua-video
```

Within 5-10 minutes:
- `segment/{id}.json` appears (dead + labeled content segments)
- `transcript/{id}.json` appears (word-level speaker turns, cached)
- `transcript/{id}.vtt` appears (WebVTT for Aviary)
- The Editor shows the segments on the interactive timeline

## Step 7: Re-Segmentation

When an archivist clicks "Re-Segment" in the Editor, the same ECS task runs again. On re-runs:
- Dead-space detection: **re-runs** (fresh probe)
- Transcription: **skipped** (loads cached `transcript/{id}.json` from S3)
- Content labeling: **re-runs** (fresh Claude calls with new boundaries)

Cost of re-segmentation: ~$0.07 (no Transcribe charge).

## Configuration

Environment variables on the ECS container (set in CDK stack):

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTENT_SEGMENT` | `auto` | `auto` = label if API key present; `off` = dead-space only |
| `TRIM_MODE` | `black` | `black` (conservative) or `static` (aggressive) |
| `MERGE_GAP` | `2.0` | Bridge dead regions closer than this (seconds) |
| `MIN_DEAD_DUR` | `3.0` | Minimum dead span duration to report (seconds) |
| `MIN_CONTENT_DUR` | `5.0` | Content spans shorter than this absorbed into dead |
| `TRANSCRIBE_LANGUAGE` | `en-US` | AWS Transcribe language code |
| `MAX_SPEAKERS` | `10` | Max speakers for diarization |
| `MAX_FRAME_WIDTH` | `768` | Downscale frames before sending to Claude |
| `CLAUDE_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID for content labeling |

## Dead-Space Detectors

| Signal | Method | Qualifies as anchor? |
|--------|--------|---------------------|
| Black frames | ffmpeg `blackdetect` | Yes |
| White frames | ffmpeg `negate` + `blackdetect` | Yes |
| Color bars (SMPTE) | OpenCV: saturation + vertical band uniformity | Yes |
| Snow/static | OpenCV: Laplacian variance + frame-to-frame MAD | Yes |
| Freeze frames | ffmpeg `freezedetect` | No (supporting only) |
| Silence | ffmpeg `silencedetect` | No (supporting only) |

A dead span must contain at least one anchor signal (black, white, bars, or snow). Freeze and silence strengthen confidence but cannot define a dead span alone.

## S3 Path Conventions

| Path | Content | Written by |
|------|---------|------------|
| `video/{name}.mp4` | Source video uploads | Frontend |
| `segment/{id}.json` | Segment timeline JSON | Backend (auto) + Frontend (edits) |
| `transcript/{id}.json` | Word-level speaker turns (cached) | Backend |
| `transcript/{id}.vtt` | WebVTT for Aviary | Backend |
| `review/{name}.txt` | Flagged-for-review markers | Backend |
| `edit/{name}_trim_request.json` | Trim instructions | Frontend |
| `edit/{name}_trimmed.mp4` | Trimmed video output | Backend (trim mode) |

## Cost

| Component | Per 1-hour video | Notes |
|-----------|-----------------|-------|
| Dead-space detection | ~$0.01 | EC2 time only |
| Transcription (first run) | ~$0.72 | Cached for re-runs |
| Content labeling | ~$0.06 | ~3 Claude Sonnet calls |
| **Total (first run)** | **~$0.79** | 5-10 minutes |
| **Total (re-segment)** | **~$0.07** | 2-5 minutes (cached transcript) |

Infrastructure idle cost: ~$33/month (NAT gateway). EC2 scales to 0 when no tasks are running.

## Instance Type

Currently `c5.xlarge` (4 vCPU, 8GB RAM, ~$0.17/hr). No GPU needed — all AI runs via Bedrock + Transcribe using the ECS task role, not local inference.

## Updating the Pipeline Code

After modifying files in `video-processing/`:

```bash
cdk deploy \
  --parameters AmplifyBucketName=amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1 \
  --profile scua-video
```

CDK rebuilds the Docker image and updates the ECS task definition. Running tasks are not affected; new tasks use the updated image.

## Troubleshooting

**ECS task not starting:**
- Check Lambda logs: `aws logs tail /aws/lambda/scua-video-trigger --follow --profile scua-video`
- Verify the file is in `video/` prefix and > 1MB

**Transcription fails:**
- Ensure the ECS task role has `transcribe:StartTranscriptionJob` permission
- Check that the video has an audio track

**Content labeling fails:**
- Check CloudWatch logs for Bedrock API errors
- Verify the ECS task role has `bedrock:InvokeModel` permission
- Pipeline continues without labels (segments output as type "I" instead of "C")

**No segments appearing in Editor:**
- Check `segment/{id}.json` exists in S3
- The Editor loads by video ID from the DynamoDB record — confirm the ID matches
