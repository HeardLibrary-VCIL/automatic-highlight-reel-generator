# Level 3 Security Hardening — HighlightProcessor

Status of the `automatic-highlight-reel-generator` stack against a Vanderbilt
University **Data Classification Level 3** baseline.

> **Scope note.** This document describes technical controls implemented in the
> CDK stack (`lib/highlight-processor-stack.ts`) plus the account/org-level items
> that must be confirmed with VU. It is **not** a compliance certification.
> Level 3 eligibility is determined by the VU Office of Cybersecurity / Office of
> Data & Strategic Analytics (DSA), not by this stack.

## Deployment posture

| Stack | Data class | Hardening | Networking |
|-------|-----------|-----------|------------|
| `HighlightProcessorDevStack` | PoC / non-L3 | OFF (default) | Own VPC, no NAT, **public IP** (PoC only) |
| `HighlightProcessorStack` (prod) | Level 3 target | ON (default) | Private subnet, VPC endpoints, **no public IP** |

The `harden` prop defaults ON for prod and OFF for the dev PoC. The dev
public-subnet/public-IP path is **not** Level 3 eligible and must never carry
Level 3 data.

## Controls implemented in code

| # | Control | Where | Requirement addressed |
|---|---------|-------|-----------------------|
| 1 | Customer-managed KMS key (CMK), key rotation on, `RETAIN` removal | `DataKey` | Encryption at rest with a key VU controls; key policy is this-account-only (no cross-account principal) |
| 2 | S3 objects read/written by the task encrypted with the CMK; task role granted scoped `EncryptDecrypt` on that key only | `dataKey.grantEncryptDecrypt(taskRole)` | At-rest encryption + least-privilege KMS |
| 3 | EBS root volume encrypted | ASG `blockDevices` `encrypted: true` | Source video on the instance encrypted at rest |
| 4 | CloudWatch log groups encrypted with the CMK | ECS + trigger Lambda log groups | Logs may echo data detail; encrypt at rest |
| 5 | Security group egress restricted to HTTPS (443) to the VPC CIDR; **no** allow-all-outbound when hardened; **no inbound rules anywhere** | `VideoProcessorSG` | Limit connectivity to only ports/services needed |
| 6 | VPC endpoints — S3 gateway + interface endpoints for ECR, ECR-Docker, CloudWatch Logs, STS, Transcribe, Bedrock runtime | `harden && !importedVpc` | Task reaches AWS services with **no internet route**; NAT can be removed |
| 7 | Private subnet, no public IP for the task | `assignPublicIp=false` (prod), private-with-egress subnets | Public-IP-by-exception rule (see below) |

## Vanderbilt public-IP rule — how this stack complies

VU policy: public IPs are **by exception only**, terminated at a load balancer with
a private-IP backend where feasible, behind a firewall, with connectivity limited
to only needed ports/services.

- **Prod (L3):** the task runs in a **private subnet with no public IP**. All egress
  is to in-VPC interface endpoints over 443. No public IP → no exception needed.
- **Dev (PoC):** uses a public IP for cost/simplicity. This is acceptable **only**
  because it holds no Level 3 data. If the PoC ever needs a public entry point
  carrying real data, that requires a **VU Cybersecurity public-IP exception** and
  should be fronted by a load balancer (ALB) with the compute on a private IP.

## Cross-account exposure — defenses in place

Cross-account access in AWS is almost always a *misconfiguration* (over-broad
resource policies, missing `ExternalId`, wildcard principals), not a platform break.
This stack defends against it by:

- CMK key policy scoped to **this account only** — no cross-account principal is
  ever added. Even if a bucket policy were later misconfigured, L3 objects stay
  unreadable without KMS access, which is granted only to the task role.
- IAM grants written with `grant*()` helpers (scoped statements), no wildcard KMS
  resource.
- No `Principal: "*"` and no bare account-root trust anywhere in the stack.

## MUST confirm / do at the account or org level (not in this stack)

These are outside the CDK stack and must be verified with VU central security:

- [ ] Account is VU-sanctioned for Level 3 data (confirmed: yes, per project owner).
- [ ] **Transcribe & Bedrock** covered under VU's AWS agreement and configured to not
      retain/train on data (confirmed approved; verify data-handling settings).
- [ ] **CloudTrail** (management + S3 data events) enabled and retained.
- [ ] **VPC Flow Logs** enabled on the prod VPC.
- [ ] **GuardDuty** enabled.
- [ ] **IAM Access Analyzer** enabled — directly flags any resource reachable from
      outside the account (the key detective control for cross-account exposure).
- [ ] **S3 Block Public Access** on at the account level; bucket-level TLS-only +
      SSE-KMS-only **bucket policy** enforced. (The video bucket is Amplify-managed
      and imported here via `fromBucketName`, so this policy must be applied on the
      bucket's owning stack or via the existing `BucketConfigHandler` custom resource.)
- [ ] **S3 access logging** enabled on the video bucket.
- [ ] **Data retention / immutability** requirement confirmed — the current stack
      *suspends* bucket versioning and expires noncurrent versions quickly, which may
      conflict with an L3 retention requirement. Reconcile with DSA.

## Multi-deployment safety

All colliding physical names are scoped to `this.stackName`, so any number of
stacks (any name, any dev/prod flavor) coexist in one account:

- ECS + Lambda log groups: `/ecs/scua-video-processor-<stackName>`, `/aws/lambda/scua-video-trigger-<stackName>`
- ECS task-definition family: `scua-video-processor-<stackName>`
- S3 notification id: `scua-video-trim-trigger-<stackName>`
- S3 lifecycle rule id: `scua-cleanup-noncurrent-<stackName>`

**Caveat:** `put_bucket_lifecycle_configuration` replaces a bucket's entire
lifecycle config, so two stacks pointed at the **same** bucket would still clobber
each other's rule regardless of id. This is safe only because dev and prod use
**separate** buckets. Renaming the log groups/task family means the next prod
deploy creates new ones and orphans the old (empty) ones — harmless, no data loss.

## Known gaps / follow-ups

- **TLS-only + KMS-only bucket policy** is not set here because the bucket is imported.
  Apply on the source (Amplify) stack, or extend `BucketConfigHandler` to `put_bucket_policy`.
- **EBS CMK association**: the block device is guaranteed *encrypted*; associating the
  specific CMK (vs the account default EBS key) requires a launch-template `kmsKeyId`.
  Confirm whether VU requires the CMK specifically for EBS or accepts the AWS-managed
  EBS key.
- **`cdk synth` verification** of the hardened prod template was blocked by expired AWS
  credentials (the container image `fromAsset` forces auth). TypeScript type-checks
  clean; run `cdk synth HighlightProcessorStack` with valid creds to emit and review
  the final template before deploy.
