"""Tests for the S3-event routing in lambda/handler.py.

Run: AWS_DEFAULT_REGION=us-east-1 python3 -m unittest discover -s lambda -p 'test_*.py'

The AWS clients are patched out, so nothing here talks to AWS. What matters is
which branch a record takes: an internal copy (how the editor renames a video)
must never be mistaken for a new upload, because that re-runs the whole pipeline
and overwrites the segment JSON the rename just moved.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

import handler  # noqa: E402


def s3_event(key, event_name, bucket="scua-video-storage"):
    return {"Records": [{
        "eventName": event_name,
        "s3": {"bucket": {"name": bucket}, "object": {"key": key}},
    }]}


class RoutingTest(unittest.TestCase):
    def setUp(self):
        # Every launch path is stubbed; the assertions are about which one runs.
        self.run_task = patch.object(handler.ecs, "run_task").start()
        self.run_task.return_value = {"tasks": [{"taskArn": "arn:aws:ecs:task/abc"}]}
        self.validate = patch.object(
            handler, "validate_video_file",
            return_value=(True, "Valid video file", {"size": 5_000_000, "content_type": "video/mp4"}),
        ).start()
        for var in ("CLUSTER_NAME", "TASK_DEFINITION", "SUBNET_IDS",
                    "SECURITY_GROUP", "ASSIGN_PUBLIC_IP", "CAPACITY_PROVIDER_NAME"):
            os.environ[var] = "test"
        self.addCleanup(patch.stopall)

    def test_a_real_upload_starts_segmentation(self):
        handler.lambda_handler(s3_event("video/RCC_183.mp4", "ObjectCreated:Put"), None)
        self.assertEqual(self.run_task.call_count, 1)

    def test_a_multipart_upload_starts_segmentation(self):
        handler.lambda_handler(
            s3_event("video/big.mp4", "ObjectCreated:CompleteMultipartUpload"), None)
        self.assertEqual(self.run_task.call_count, 1)

    def test_a_rename_copy_does_not_start_segmentation(self):
        # The destination of a rename: without this guard the pipeline re-runs and
        # overwrites segment/Fisk_Jubilee.json, losing the user's manual edits.
        result = handler.lambda_handler(
            s3_event("video/Fisk_Jubilee.mp4", "ObjectCreated:Copy"), None)
        self.run_task.assert_not_called()
        body = result["body"] if isinstance(result, dict) else str(result)
        self.assertIn("Internal copy", body)

    def test_a_copy_never_reaches_video_validation(self):
        handler.lambda_handler(s3_event("video/Copied.mp4", "ObjectCreated:Copy"), None)
        self.validate.assert_not_called()

    def test_a_trim_request_still_routes_to_trim_mode(self):
        with patch.object(handler, "launch_trim_task", return_value={"key": "k"}) as launch:
            handler.lambda_handler(
                s3_event("edit/abc_trim_request.json", "ObjectCreated:Put"), None)
            launch.assert_called_once()

    def test_a_segment_request_still_routes_to_detect_mode(self):
        with patch.object(handler, "launch_segment_task", return_value={"key": "k"}) as launch:
            handler.lambda_handler(
                s3_event("edit/abc_segment_request.json", "ObjectCreated:Put"), None)
            launch.assert_called_once()

    def test_non_trigger_prefixes_are_skipped(self):
        handler.lambda_handler(s3_event("segment/RCC_183.json", "ObjectCreated:Put"), None)
        self.run_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
