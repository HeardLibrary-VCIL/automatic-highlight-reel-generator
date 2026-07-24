"""Shared Amazon Bedrock Claude client for the whole segmentation pipeline.

Every model call -- the VLM frame classification (segment_content / segment_shots)
AND the transcript labeling (segment_label) -- goes through Bedrock so it reuses
the project's AWS credentials (the same `.env` transcribe.py uses) instead of a
separate ANTHROPIC_API_KEY.

Account note (337513903342, us-east-1): only Haiku 4.5 is currently enabled in
Bedrock -- `us.anthropic.claude-opus-4-8` / `claude-sonnet-5` return 403
"not available for this account". The id also NEEDS the `us.` cross-region
inference-profile prefix; the bare `anthropic.claude-haiku-4-5-...` returns 400.
Request Bedrock model access for a stronger model and bump BEDROCK_MODEL if the
vision labels are weak.

Run with the same AWS env as transcribe.py:
    export AWS_SHARED_CREDENTIALS_FILE=../.env AWS_PROFILE=337513903342_PowerUserAccess
"""

BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def make_client(region: str = BEDROCK_REGION):
    """An AnthropicBedrock client. Exposes the same messages.create surface as
    anthropic.Anthropic(), so callers only swap the constructor."""
    from anthropic import AnthropicBedrock
    return AnthropicBedrock(aws_region=region)
