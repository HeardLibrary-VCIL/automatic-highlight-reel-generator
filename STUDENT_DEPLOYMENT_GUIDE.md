## Deployment

### Prerequisites

1. AWS account with CDK bootstrapped
2. AWS CLI configured with a profile (e.g., `scua-vcil`)
3. Node.js 18+ and npm
4. Docker running locally (for building the ECS container image)
5. Claude Sonnet 4.6 enabled in Amazon Bedrock (Console → Bedrock → Model access)

### Two stacks

This app defines **two** stacks, so every `cdk` command must name the one you want
(a bare `cdk deploy` errors with "specify which stacks to use"):

| Stack | Use | Compute | Networking | Hardening |
|-------|-----|---------|------------|-----------|
| `HighlightProcessorDevStack` | Dev / PoC / **student sandboxes** | Fargate (pay-per-task, no idle cost) | NAT-free public subnet | OFF |
| `HighlightProcessorStack` | Production (Level 3) | EC2 / ASG | Private subnet + VPC endpoints | ON |

> **Students: always use the DEV stack.** It is serverless (no idle EC2 cost),
> NAT-free (~$0 standing network cost)

---

## Deploy your own dev stack (students)

Each student can deploy an **isolated** copy of the dev pipeline in the same AWS
account. All resource names are scoped to the stack name, so your deployment won't
collide with anyone else's — as long as you (1) pick a **unique stack name** and
(2) point at **your own** Amplify bucket.

### 1. One-time setup

```bash
# Node 18+ and Docker must be installed, and Docker must be running.
node --version        # expect v18+ (repo tested on v20)
docker info           # must succeed — the ECS image is built locally

cd automatic-highlight-reel-generator
npm install

# Confirm your AWS profile works and note the account id.
aws sts get-caller-identity --profile <YOUR_PROFILE_NAME>
```

If the account/region has never been used with CDK, bootstrap it once (safe to
re-run; it's a no-op if already done):

```bash
npx cdk bootstrap --profile <YOUR_PROFILE_NAME>
```

### 2. Find your Amplify bucket name

Use the bucket from **your own** SCUA-Video-Editing Amplify sandbox
(`amplify_outputs.json` → `storage.bucket_name`). Do **not** reuse a teammate's
bucket — two stacks sharing one bucket fight over its S3 notification/lifecycle
config.

### 3. Deploy under your own stack name

Pass a unique suffix via `-c devSuffix=<you>` (no code edits needed). The suffix
becomes both the CloudFormation stack name and the namespace on every scoped
resource, so your deployment is fully isolated from your classmates':

```bash
npx cdk deploy HighlightProcessorDevStack-jane \
  -c devSuffix=jane \
  --parameters AmplifyBucketName=<your-amplify-bucket-name> \
  --profile <YOUR_PROFILE_NAME>
```

Use the same suffix for the stack name argument and the `devSuffix` context value.
Pick something unique (your name/VUnetID). With this suffix, your resources are:

- `/ecs/scua-video-processor-HighlightProcessorDevStack-jane`
- `/aws/lambda/scua-video-trigger-HighlightProcessorDevStack-jane`
- task family `scua-video-processor-HighlightProcessorDevStack-jane`
- S3 notification id `scua-video-trim-trigger-HighlightProcessorDevStack-jane`

> First run builds and pushes the ECS container image, so expect a few minutes.
> Docker must be running locally.

### 4. Verify it worked

```bash
# The stack should show CREATE_COMPLETE / UPDATE_COMPLETE
aws cloudformation describe-stacks \
  --stack-name HighlightProcessorDevStack-jane \
  --query 'Stacks[0].StackStatus' --profile s<YOUR_PROFILE_NAME>

# Upload a test video to trigger the pipeline
aws s3 cp test.mp4 s3://<your-amplify-bucket-name>/video/test.mp4 --profile <YOUR_PROFILE_NAME>

# Watch the task logs (name uses YOUR stack name)
aws logs tail /ecs/scua-video-processor-HighlightProcessorDevStack-jane \
  --follow --profile <YOUR_PROFILE_NAME>
```

### 5. Tear it down when you're done

Fargate has no idle cost, but clean up to keep the account tidy:

```bash
npx cdk destroy HighlightProcessorDevStack-jane -c devSuffix=jane --profile <YOUR_PROFILE_NAME>
```

> Pass the same `-c devSuffix=jane` you deployed with, so CDK resolves the same
> stack id.

> Note: the dev stack is unhardened, so it has **no** retained KMS key — destroy
> removes everything cleanly. (The prod stack retains its CMK by design.)

### Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `specify which stacks to use` | You omitted the stack name — add `HighlightProcessorDevStack-<you>`. |
| `security token ... is expired` | Refresh creds: re-run your SSO/`aws sso login`, then `aws sts get-caller-identity --profile scua-vcil`. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop; the ECS image builds locally. |
| `... already exists` on a log group | Another stack uses the same name — pick a more unique stack name. |
| `This stack uses assets, so the toolkit stack must be deployed` | Run `npx cdk bootstrap --profile scua-vcil` once. |
| Bedrock `AccessDenied` on `InvokeModel` | Enable Claude model access in the Bedrock console for your account/region. |

### What Gets Created

Both stacks create:
- Docker image built from `video-processing/` and pushed to ECR
- Lambda trigger for S3 events
- IAM roles (task role: S3 + Transcribe + Bedrock; execution role: ECR + logs)
- CloudWatch log groups, scoped to the stack name (`/ecs/scua-video-processor-<stackName>`)
- S3 notification config + bucket lifecycle rule on the Amplify bucket

Differences by stack:

| | `HighlightProcessorDevStack` (dev/students) | `HighlightProcessorStack` (prod) |
|---|---|---|
| Compute | Fargate (no idle cost) | ECS cluster + c5.xlarge ASG (scales to 0) |
| Networking | VPC, NAT-free public subnet | VPC, private subnet + VPC endpoints |
| NAT gateway | none | 1 (unless attached to a central VPC via `-c vpcId`) |
| Encryption | account-default | customer-managed KMS key (CMK) for S3/EBS/logs |

See [SECURITY_HARDENING_L3.md](SECURITY_HARDENING_L3.md) for the prod hardening details.

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

