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
