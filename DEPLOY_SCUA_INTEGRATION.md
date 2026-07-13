# SCUA Video Trimmer — Deployment Guide

## Overview

This deploys the dead-space trimmer as a backend processing pipeline for the SCUA-Video-Editing Amplify app. When a video is uploaded through the frontend to `video/`, it automatically:

1. Detects head/tail dead space (black frames, color bars, snow, silence)
2. Trims the video → uploads to `edit/{basename}_trimmed.mp4`
3. Writes auto-detected segments → `segment/{basename}.json`
4. The SCUA Editor auto-loads these segments for review/editing

## Architecture

```
SCUA Frontend (Amplify)
    │ upload to video/*
    ▼
S3 Bucket (Amplify-managed)
    │ S3 event notification (video/ prefix)
    ▼
Lambda (validate + launch ECS)
    │
    ▼
ECS Task (g4dn.2xlarge GPU, ASG min 0)
    │ ffmpeg blackdetect/freezedetect/silencedetect
    │ + color bar + snow detection
    ▼
S3: edit/{name}_trimmed.mp4 + segment/{name}.json
    │
    ▼
SCUA Editor (auto-loads segments, shows "Trimmed" badge)
```

## Prerequisites

- AWS CDK installed (`npm install -g aws-cdk`)
- Docker running
- Hugging Face token (for future AI content search features)
- The SCUA-Video-Editing Amplify app deployed (you need the bucket name)

## Instance Type

The stack currently uses `c5.xlarge` (CPU-only, 4 vCPU, 8GB RAM, ~$0.17/hr) for the dead-space trimmer. This requires no GPU quota.

**To upgrade to GPU when quota is approved**, change in `lib/highlight-processor-stack.ts`:

```ts
// CPU (current):
instanceType: ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.XLARGE),
machineImage: ecs.EcsOptimizedImage.amazonLinux2023(),
// container: memoryLimitMiB: 7168, cpu: 4096

// GPU (when ready):
instanceType: ec2.InstanceType.of(ec2.InstanceClass.G4DN, ec2.InstanceSize.XLARGE2),
machineImage: ecs.EcsOptimizedImage.amazonLinux2023(ecs.AmiHardwareType.GPU),
// container: memoryLimitMiB: 30720, cpu: 8192, gpuCount: 1
```

GPU requires: Service Quotas → EC2 → "Running On-Demand G and VT instances" ≥ 8 vCPUs.

## Step 1: Get the Amplify Bucket Name

From the SCUA-Video-Editing project:
```bash
cat amplify_outputs.json | jq -r '.storage.bucket_name'
```

This returns something like: `amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1`

## Step 2: Install CDK Dependencies

```bash
cd Project2/automatic-highlight-reel-generator
npm install
```

## Step 3: Bootstrap CDK (first time only)

```bash
cdk bootstrap aws://ACCOUNT_ID/us-east-1
```

## Step 4: Deploy

**Option A: Dead-space detection only (no AI model)**

```bash
cdk deploy --parameters AmplifyBucketName=amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1
```

**Option B: With VLM content search (PaliGemma model)**

Requires a Hugging Face account with access to `google/paligemma2-3b-mix-224`. The token is used during Docker build to download the ~6GB model into the container image.

```bash
export HUGGINGFACE_TOKEN='hf_your_token_here'

cdk deploy --parameters AmplifyBucketName=amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1
```

Note: Option B significantly increases Docker build time (15-30 min on first build) and the resulting container image size (~8GB). The model is baked into the image so no runtime download is needed.


This creates:
- VPC with NAT gateway
- ECS Cluster with GPU ASG (min 0, max 2)
- Lambda trigger listening on the Amplify bucket's `video/` prefix
- S3 notification on the existing Amplify bucket (cross-stack)
- IAM roles with least-privilege access

## Step 5: Test

Upload a video through the SCUA frontend. Within ~2-3 minutes:
- Check CloudWatch logs: `/ecs/scua-video-processor`
- The trimmed video appears in `edit/`
- Segment JSON appears in `segment/`
- The VideoSelect page shows "Segmented" and "Trimmed" badges

## Configuration

Environment variables on the ECS task (set in CDK stack):

| Variable | Default | Description |
|----------|---------|-------------|
| `RESULT_PREFIX` | `edit` | S3 prefix for trimmed videos |
| `SEGMENT_PREFIX` | `segment` | S3 prefix for segment JSON |
| `TRIM_MODE` | `black` | `black` (conservative) or `static` (aggressive) |
| `REVIEW_POLICY` | `upload_original` | What to do with uncertain files |

## Cost

- **Idle:** ~$33/month (NAT gateway)
- **Per video:** ~$0.02-0.05 (trimmer runs in 30-90s, no GPU needed for ffmpeg-only)
- **Tip:** The trimmer doesn't need the GPU. Future optimization: switch to a cheaper instance type (c5.xlarge) for trim-only and reserve GPU instances for AI content search.

## S3 Path Conventions

| Path | Content | Written by |
|------|---------|------------|
| `video/{name}.mp4` | Source uploads | Frontend (Upload page) |
| `edit/{name}_trimmed.mp4` | Trimmed videos | ECS trimmer |
| `segment/{name}.json` | Segment annotations | ECS trimmer (auto) + Frontend (manual edits) |
| `review/{name}.txt` | Flagged-for-review markers | ECS trimmer |

Changes to adapt to deadspace trimming only:

c5.xlarge (4 vCPU, 8GB RAM, ~$0.17/hr) — no GPU quota needed, general compute quota is typically available by default
Removed GPU AMI (uses standard ECS-optimized AMI)
Removed gpuCount: 1 from container
Adjusted memory/CPU to match c5.xlarge
When your G4DN quota comes through, just revert these two changes (instance type + container resources) and redeploy.

aws s3 cp "s3://scua-video/10/Web Copy/RCC_10.mp4" "s3://amplify-d3f0pl9vo50wn1-ma-scuavideostoragebucket58-9nitsvxrhdt1/video/RCC_10.mp4"
