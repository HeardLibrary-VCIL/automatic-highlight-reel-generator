"""Shared Amazon Bedrock Claude client for the segmentation pipeline.

Uses AnthropicBedrock which authenticates via the ECS task role (IAM) — no API
key needed. The model ID uses the `us.` cross-region inference profile prefix.

Account 337513903342, us-east-1. Current active model: Claude Sonnet 4.6.
Override via CLAUDE_MODEL env var if a newer model becomes available.

For local dev, set AWS credentials:
    export AWS_SHARED_CREDENTIALS_FILE=../.env AWS_PROFILE=337513903342_PowerUserAccess
"""

import os

BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL = os.environ.get("CLAUDE_MODEL", "us.anthropic.claude-sonnet-4-6")


def make_client(region: str = BEDROCK_REGION):
    """An AnthropicBedrock client. Exposes the same messages.create surface as
    anthropic.Anthropic(), so callers only swap the constructor."""
    from anthropic import AnthropicBedrock
    return AnthropicBedrock(aws_region=region)


def create_with_retry(client, *, max_retries=6, base_delay=2.0, **kwargs):
    """client.messages.create with exponential backoff on Bedrock throttling.

    A long video produces one vision call per shot; without backoff a burst of
    calls trips Bedrock's rate limit (HTTP 429) and the whole segmentation aborts
    to dead-space-only. Retries on 429 / throttling / overloaded with exponential
    backoff + jitter so labeling completes instead of failing the run."""
    import time
    import random

    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            status = getattr(e, "status_code", None)
            throttled = (status == 429 or "429" in msg or "too many requests" in msg
                         or "throttl" in msg or "overloaded" in msg
                         or "rate" in msg and "limit" in msg)
            if not throttled or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
            time.sleep(delay)
