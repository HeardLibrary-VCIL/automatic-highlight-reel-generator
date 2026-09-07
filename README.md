# SCUA Video Segment Detector

Automated dead-space detection and content labeling for archival video collections. Built for the Vanderbilt Special Collections and University Archives (SCUA) to accelerate the processing of digitized VHS tapes and other archival recordings.

| Section | Description |
|:--------|:------------|
| [Overview](#overview) | What this system does and why |
| [Pipeline](#pipeline) | The 4-stage processing pipeline |
| [Architecture](#architecture) | AWS infrastructure and data flow |
| [Deployment](#deployment) | How to deploy the backend |
| [Configuration](#configuration) | Environment variables and tuning |
| [S3 Layout](#s3-layout) | Bucket prefix conventions |
| [Cost](#cost) | Per-video processing cost |
| [Frontend Integration](#frontend-integration) | How the SCUA Editor consumes results |
| [Credits](#credits) | Team and acknowledgments |
| [License](#license) | MIT License |

---

## Overview

Archival video collections — digitized VHS tapes, legacy recordings, institutional footage — arrive as raw files that may contain hours of mixed content separated by dead space (black frames, color bars, static/snow, white frames, or silence). Before these videos can be cataloged, discovered, or cited, an archivist must identify where content begins and ends, and describe what each segment contains.

This system automates that process:

1. **Detects dead space throughout the entire video** — not just at the head and tail, but internal gaps where recording stopped and restarted (black, white, bars, snow, freeze+silence).
2. **Transcribes the audio** with speaker diarization (AWS Transcribe), producing word-level timestamps.
3. **Labels each content span** with a short description using Claude Sonnet 4.6 via Amazon Bedrock (frame + transcript context).
4. **Writes structured segment JSON** that the SCUA Editor UI displays on an interactive timeline for archivist review and correction.

No GPU required. The pipeline runs on CPU-only ECS tasks (c5.xlarge), with AI classification offloaded to Amazon Bedrock (Claude). Authentication is via the ECS task role — no API keys to manage.

---

## Pipeline

When a video is uploaded to S3 (`video/` prefix), the system automatically processes it through 4 stages:

### Stage 1: Full-Video Dead-Space Detection

Uses ffmpeg and OpenCV to probe the entire video for dead regions:

| Detector | Signal | Method |
|----------|--------|--------|
| Black frames | Near-black sustained frames | ffmpeg `blackdetect` |
| White frames | Near-white sustained frames | ffmpeg `negate` + `blackdetect` |
| Color bars | SMPTE/EBU test patterns | OpenCV: saturation + vertical band uniformity |
| Video static (snow) | Tape run-out noise | OpenCV: Laplacian variance + frame-to-frame MAD |
| Freeze frames | Unchanging video (supporting signal) | ffmpeg `freezedetect` |
| Silence | Near-silent audio (supporting signal) | ffmpeg `silencedetect` |

Dead regions are merged (bridging gaps < 2s) and filtered (minimum 3s duration). Only regions anchored by black, white, bars, or snow qualify as dead spans — freeze and silence alone are not sufficient (they catch static camera shots in real program otherwise).

Output: alternating dead spans + content spans covering the full video duration.

### Stage 2: Transcription (Cached)

AWS Transcribe reads the video directly from S3 (no re-upload) with speaker diarization enabled. Returns word-level timestamps grouped into speaker turns.

The transcript is **cached** to `transcript/{name}.json` and `transcript/{name}.vtt` (WebVTT for Aviary). On re-segmentation runs, the cached transcript is reused — Transcribe is not re-run.

### Stage 3: Content Labeling (Claude Sonnet 4.6 via Bedrock)

For each content span between dead gaps:
- Samples a frame from the midpoint
- Includes a transcript excerpt for context
- Asks Claude to describe the segment and provide a short title via Amazon Bedrock
- ~1 API call per content span (~$0.06 per hour of video)
- Authenticates via ECS task role (IAM) — no API key needed

Graceful fallback: if labeling fails, content spans are still output with their transcripts but without AI-generated titles.

### Stage 4: Write Segment JSON

Writes `segment/{name}.json` to S3 :

```json
{
  "video": "RCC_183",
  "duration": 1847.5,
  "dead_span_count": 3,
  "content_span_count": 2,
  "segments": [
    {"segment_start": 0, "segment_end": 45.2, "segment_type": "D", "title": "Dead space (bars)"},
    {"segment_start": 45.2, "segment_end": 892.1, "segment_type": "C", "title": "Campus tour interview",
     "transcript": "Welcome to the archives today...", "description": "Two people walking..."},
    {"segment_start": 892.1, "segment_end": 904.8, "segment_type": "D", "title": "Dead space (black)"},
    {"segment_start": 904.8, "segment_end": 1802.3, "segment_type": "C", "title": "Lecture recording",
     "transcript": "Good afternoon everyone...", "description": "A speaker at a podium..."},
    {"segment_start": 1802.3, "segment_end": 1847.5, "segment_type": "D", "title": "Dead space (snow)"}
  ]
}
```

---

## Architecture

```
Upload video to S3 (video/*.mp4)
  → [S3 event notification]
  → Lambda (validates file, launches ECS task)
  → ECS Task (c5.xlarge, CPU-only):
      1. ffmpeg + OpenCV dead-space probe (full video)
      2. AWS Transcribe (reads from S3, cached)
      3. Claude Sonnet 4.6 labeling via Bedrock (1 call per content span)
      4. Write segment JSON + transcript JSON/VTT to S3
  → SCUA Editor UI (React/Amplify) reads segment/{id}.json
  → Archivist reviews, edits, approves
  → [Optional] Trim request → ECS task (MODE=trim) → trimmed .mp4
```

Infrastructure (CDK):
- **VPC** with NAT gateway (ECS tasks need internet for Transcribe + Bedrock)
- **ECS Cluster** with Auto Scaling Group (c5.xlarge, min 0 / max 2)
- **Lambda** trigger on S3 `video/*` uploads and `edit/*_trim_request.json`
- **IAM roles** with Bedrock InvokeModel + Marketplace permissions
- **Custom resources** for S3 notification config + bucket lifecycle

---

## Deployment

### Prerequisites

1. AWS account with CDK bootstrapped
2. AWS CLI configured with a profile (e.g., `scua-vcil`)
3. Node.js 18+ and npm
4. Docker running locally (for building the ECS container image)
5. Claude Sonnet 4.6 enabled in Amazon Bedrock (Console → Bedrock → Model access)

### Deploy

```bash
cd automatic-highlight-reel-generator
npm install
cdk deploy \
  --parameters AmplifyBucketName=<your-amplify-bucket-name> \
  --profile scua-video
```

Find the bucket name in `amplify_outputs.json` → `storage.bucket_name` in the SCUA-Video-Editing project.

### What Gets Created

- VPC (2 AZs, 1 NAT gateway)
- ECS cluster + c5.xlarge ASG (scales to 0 when idle)
- Docker image built from `video-processing/` and pushed to ECR
- Lambda trigger for S3 events
- IAM roles (task role: S3 + Transcribe; execution role: ECR + Secrets Manager)
- CloudWatch log group (`/ecs/scua-video-processor`)
- S3 notification config on the Amplify bucket
- Bucket lifecycle rule (noncurrent version cleanup)

---

## Configuration

Environment variables set on the ECS container (configurable in the CDK stack):

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTENT_SEGMENT` | `auto` | `auto` = label if API key present; `off` = dead-space only |
| `TRIM_MODE` | `black` | `black` (conservative) or `static` (aggressive) |
| `MERGE_GAP` | `2.0` | Bridge dead regions closer than this (seconds) |
| `MIN_DEAD_DUR` | `3.0` | Minimum duration for a dead span (seconds) |
| `MIN_CONTENT_DUR` | `5.0` | Content spans shorter than this get absorbed |
| `TRANSCRIBE_LANGUAGE` | `en-US` | AWS Transcribe language code |
| `MAX_SPEAKERS` | `10` | Max speakers for diarization |
| `MAX_FRAME_WIDTH` | `768` | Downscale frames to this width before sending to Claude |
| `CLAUDE_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID for content labeling |

---

## S3 Layout

All paths are within the Amplify-managed bucket:

| Prefix | Contents | Written by |
|--------|----------|------------|
| `video/` | Source video uploads (.mp4) | Frontend (user upload) |
| `segment/{id}.json` | Segment timeline JSON | Backend (ECS) |
| `transcript/{id}.json` | Word-level speaker turns (cached) | Backend (ECS) |
| `transcript/{id}.vtt` | WebVTT for Aviary | Backend (ECS) |
| `review/{name}.txt` | Review markers for flagged videos | Backend (ECS) |
| `edit/{name}_trim_request.json` | Trim instructions from Editor | Frontend |
| `edit/{name}_trimmed.mp4` | Trimmed video output | Backend (ECS, trim mode) |

---

## Cost

Per 1-hour video:

| Stage | Cost | Time |
|-------|------|------|
| Dead-space detection (ffmpeg + OpenCV) | ~$0.01 (EC2 time) | 2-5 min |
| Transcription (AWS Transcribe) | ~$0.72 | 3-5 min |
| Content labeling (Claude Sonnet 4.6 via Bedrock, ~3 calls) | ~$0.06 | 10-15 sec |
| **Total** | **~$0.79** | **5-10 min** |

Re-segmentation (cached transcript): ~$0.07, 2-5 min.

EC2 cost: c5.xlarge at $0.17/hr, scales to 0 when idle. NAT gateway: ~$0.045/hr while tasks run.

---

## Frontend Integration

The SCUA Editor (`Project2/SCUA-Video-Editing`) consumes the backend output:

- Reads `segment/{id}.json` on the Editor page to display the timeline
- Reads `transcript/{id}.json` to auto-adjust transcript text when segment boundaries are moved
- Writes `edit/{id}_trim_request.json` when the user clicks "Trim Video"
- Supports re-segmentation (triggers a fresh ECS run via the same S3 notification path)

The Editor displays:
- Color-coded timeline with D (dead) and C (content) segments
- Editable titles, transcripts, and boundaries per segment
- Skip-player that jumps over deleted segments during playback
- Download: JSON, VTT, transcript, and video

---

## Credits

**SCUA Video Segment Detector** is developed by the **Vanderbilt Cloud Innovation Lab** in partnership with the **Vanderbilt Special Collections and University Archives** and **Amazon Web Services**.

This project builds on the infrastructure originally created for the [Automatic Highlight Reel Generator](https://github.com/HeardLibrary-VCIL/automatic-highlight-reel-generator) (University of Pittsburgh Cloud Innovation Center).

---

## License

This project is distributed under the [MIT License](LICENSE).
