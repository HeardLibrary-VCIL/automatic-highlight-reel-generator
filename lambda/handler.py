import os
import boto3
import json
import logging
from urllib.parse import unquote_plus
from pathlib import Path

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ecs = boto3.client('ecs')
s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')


def _stem_from_key(key):
    """'video/RCC_10.mp4' -> 'RCC_10' (the row id convention the frontend uses)."""
    return Path(key).name.rsplit('.', 1)[0]


def ensure_video_row(bucket, key, file_info):
    """Create a DynamoDB Video row for a video/* object if one doesn't already exist.

    Covers videos copied DIRECTLY into the bucket (bypassing the app's Upload flow,
    which is what normally writes the row): the site lists videos from DynamoDB, so
    without a row a directly-copied video is processed but never appears in the UI.

    Idempotent: a conditional PutItem with attribute_not_exists(id) no-ops when a row
    already exists (e.g. the app upload created it, or a prior copy did). Best-effort —
    a failure here must not stop segmentation, so it never raises.

    Row shape mirrors src/pages/Upload.tsx: id = filename stem, displayName = original
    filename, s3Key = the object key, videoType 'original', status 'ready'. Amplify
    system fields (__typename, createdAt, updatedAt) are set so the Amplify client can
    deserialize the row.
    """
    table = os.environ.get('VIDEO_TABLE_NAME', '').strip()
    if not table:
        logger.warning("VIDEO_TABLE_NAME not set — skipping DynamoDB row creation for %s", key)
        return

    row_id = _stem_from_key(key)
    # Prefer the original filename stamped as object metadata; fall back to the key.
    meta = (file_info or {}).get('metadata') or {}
    display_name = meta.get('original-filename') or Path(key).name
    size_bytes = int((file_info or {}).get('size') or 0)
    now = _iso_now()

    item = {
        'id': {'S': row_id},
        '__typename': {'S': 'Video'},
        'displayName': {'S': display_name},
        's3Key': {'S': key},
        'videoType': {'S': 'original'},
        'status': {'S': 'ready'},
        'sizeBytes': {'N': str(size_bytes)},
        'createdAt': {'S': now},
        'updatedAt': {'S': now},
    }

    try:
        dynamodb.put_item(
            TableName=table,
            Item=item,
            # Only insert when no row with this id exists — the "if it doesn't already
            # exist" requirement. An existing row (from the app upload) is left as-is.
            ConditionExpression='attribute_not_exists(id)',
        )
        logger.info("Created DynamoDB Video row id=%s for %s", row_id, key)
    except dynamodb.exceptions.ConditionalCheckFailedException:
        logger.info("DynamoDB Video row id=%s already exists — leaving as-is", row_id)
    except Exception as e:
        # Best-effort: never block processing on the bookkeeping row.
        logger.error("Failed to create DynamoDB Video row for %s: %s", key, e)


def _iso_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _run_processor_task(container_env):
    """Start the video-processor ECS task with the given container env overrides.

    Launch-type agnostic: if LAUNCH_TYPE is set (e.g. 'FARGATE' — the dev stack),
    launch with that; otherwise fall back to CAPACITY_PROVIDER_NAME (the EC2/ASG
    prod stack). Both stacks share this handler, so it must support either.
    Returns the raw ecs.run_task response."""
    cluster = os.environ['CLUSTER_NAME']
    task_definition = os.environ['TASK_DEFINITION']
    subnet_ids = os.environ['SUBNET_IDS'].split(',')
    security_group = os.environ['SECURITY_GROUP']
    assign_public_ip = os.environ['ASSIGN_PUBLIC_IP']

    kwargs = {
        'cluster': cluster,
        'taskDefinition': task_definition,
        'count': 1,
        'networkConfiguration': {
            'awsvpcConfiguration': {
                'subnets': subnet_ids,
                'assignPublicIp': assign_public_ip,
                'securityGroups': [security_group],
            }
        },
        'overrides': {
            'containerOverrides': [{
                'name': 'video-processor',
                'environment': container_env,
            }]
        },
    }

    launch_type = os.environ.get('LAUNCH_TYPE', '').strip()
    if launch_type:
        # Fargate (dev): serverless, pay-per-task, no ASG/instances.
        kwargs['launchType'] = launch_type
    else:
        # EC2/ASG (prod): route onto the capacity provider.
        kwargs['capacityProviderStrategy'] = [
            {'capacityProvider': os.environ['CAPACITY_PROVIDER_NAME'], 'weight': 1},
        ]

    return ecs.run_task(**kwargs)

# Supported video file extensions
SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', 
    '.m4v', '.3gp', '.ogv', '.ts', '.mts', '.m2ts'
}

# Minimum file size (in bytes) to consider as a valid video
MIN_VIDEO_SIZE_BYTES = 1024 * 1024  # 1 MB

def is_video_file(filename):
    """
    Check if the file has a supported video extension
    """
    file_path = Path(filename.lower())
    return file_path.suffix in SUPPORTED_VIDEO_EXTENSIONS

def validate_video_file(bucket, key):
    """
    Validate that the S3 object is a legitimate video file
    Returns: (is_valid, reason, file_info)
    """
    try:
        # Get object metadata
        response = s3.head_object(Bucket=bucket, Key=key)
        
        # Check file size
        file_size = response.get('ContentLength', 0)
        if file_size < MIN_VIDEO_SIZE_BYTES:
            return False, f"File too small ({file_size} bytes). Minimum size: {MIN_VIDEO_SIZE_BYTES} bytes", None
        
        # Check file extension
        if not is_video_file(key):
            return False, f"Unsupported file extension. Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}", None
        
        # Check content type if available
        content_type = response.get('ContentType', '').lower()
        if content_type and not (content_type.startswith('video/') or content_type == 'application/octet-stream'):
            logger.warning(f"Unexpected content type: {content_type}. Proceeding anyway based on file extension.")
        
        file_info = {
            'size': file_size,
            'content_type': content_type,
            'last_modified': response.get('LastModified'),
            'metadata': response.get('Metadata', {})
        }
        
        return True, "Valid video file", file_info
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}", None

def launch_trim_task(bucket, trim_request_key):
    """Launch an ECS task in TRIM mode using the trim request JSON."""
    try:
        # Read the trim request to get the video key
        response = s3.get_object(Bucket=bucket, Key=trim_request_key)
        trim_data = json.loads(response['Body'].read().decode('utf-8'))
        video_key = trim_data.get('video_key', '')
        
        if not video_key:
            logger.error(f"Trim request missing video_key: {trim_request_key}")
            return {'key': trim_request_key, 'reason': 'Missing video_key'}

        # Start ECS task in TRIM mode
        response = _run_processor_task([
            {'name': 'S3_BUCKET', 'value': bucket},
            {'name': 'S3_KEY', 'value': video_key},
            {'name': 'TRIM_REQUEST_KEY', 'value': trim_request_key},
            {'name': 'MODE', 'value': 'trim'},
        ])

        if response.get('failures'):
            failure = response['failures'][0]
            logger.error(f"ECS trim task failed: {failure.get('reason')}")
            return {'key': trim_request_key, 'reason': f"ECS failed: {failure.get('reason')}"}

        task_arn = response['tasks'][0]['taskArn']
        logger.info(f"Started ECS trim task: {task_arn}")
        return {'key': trim_request_key, 'task_arn': task_arn, 'mode': 'trim'}

    except Exception as e:
        logger.error(f"Error launching trim task: {e}")
        return {'key': trim_request_key, 'reason': str(e)}


def launch_segment_task(bucket, segment_request_key):
    """Launch an ECS task in DETECT mode (re-segmentation) using the segment request JSON."""
    try:
        # Read the segment request to get the video key
        response = s3.get_object(Bucket=bucket, Key=segment_request_key)
        seg_data = json.loads(response['Body'].read().decode('utf-8'))
        video_key = seg_data.get('video_key', '')

        if not video_key:
            logger.error(f"Segment request missing video_key: {segment_request_key}")
            return {'key': segment_request_key, 'reason': 'Missing video_key'}

        # Start ECS task in detect mode (re-segmentation)
        response = _run_processor_task([
            {'name': 'S3_BUCKET', 'value': bucket},
            {'name': 'S3_KEY', 'value': video_key},
            {'name': 'MODE', 'value': 'detect'},
        ])

        if response.get('failures'):
            failure = response['failures'][0]
            logger.error(f"ECS segment task failed: {failure.get('reason')}")
            return {'key': segment_request_key, 'reason': f"ECS failed: {failure.get('reason')}"}

        task_arn = response['tasks'][0]['taskArn']
        logger.info(f"Started ECS segment task: {task_arn}")
        return {'key': segment_request_key, 'task_arn': task_arn, 'mode': 'detect'}

    except Exception as e:
        logger.error(f"Error launching segment task: {e}")
        return {'key': segment_request_key, 'reason': str(e)}


def lambda_handler(event, context):
    """
    Lambda function triggered by S3 uploads to start ECS video processing task
    Only processes valid video files.
    """

    # Log the entire event for debugging purposes to confirm invocation
    logger.info(f"Lambda triggered. Event: {json.dumps(event)}")

    processed_files = []
    skipped_files = []

    try:
        # Parse S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])
            event_name = record.get('eventName', '')

            logger.info(f"Processing file: s3://{bucket}/{key} ({event_name})")

            # A rename in the editor moves a video by copying it to the new key
            # (SCUA-Video-Editing src/utils/rename.ts). Treating that copy as a new
            # upload would re-run the whole pipeline and overwrite the segment JSON
            # the rename just moved, losing the user's manual edits. The bucket
            # notification already excludes Copy; this also holds if that config
            # drifts or a legacy ObjectCreated:* rule is still in place.
            if event_name.startswith('ObjectCreated:Copy'):
                logger.info(f"Skipping {key}: internal copy (rename), not a new upload")
                skipped_files.append({'key': key, 'reason': 'Internal copy, not a new upload'})
                continue
            
            # Route: edit/*_trim_request.json → trim mode
            if key.startswith('edit/') and key.endswith('_trim_request.json'):
                logger.info(f"Trim request detected: {key}")
                processed_files.append(launch_trim_task(bucket, key))
                continue
            
            # Route: edit/*_segment_request.json → re-segmentation mode
            if key.startswith('edit/') and key.endswith('_segment_request.json'):
                logger.info(f"Segment request detected: {key}")
                processed_files.append(launch_segment_task(bucket, key))
                continue
            
            # Route: video/*.mp4 → segment detection mode
            if not key.startswith('video/'):
                logger.info(f"Skipping {key}: not a recognized trigger pattern")
                skipped_files.append({'key': key, 'reason': 'Not a recognized trigger pattern'})
                continue
            
            # Validate that this is a video file
            is_valid, reason, file_info = validate_video_file(bucket, key)
            
            if not is_valid:
                logger.warning(f"Skipping file {key}: {reason}")
                skipped_files.append({
                    'key': key,
                    'reason': reason
                })
                continue
            
            logger.info(f"Valid video file detected: {key} ({file_info['size']} bytes, {file_info['content_type']})")

            # Ensure a DynamoDB Video row exists (creates one for videos copied
            # directly into the bucket; no-ops when the app upload already made it).
            ensure_video_row(bucket, key, file_info)

            # Start ECS task (segment-detection mode)
            response = _run_processor_task([
                {'name': 'S3_BUCKET', 'value': bucket},
                {'name': 'S3_KEY', 'value': key},
            ])
            
            # Check for failures from the API call and log them clearly
            if response.get('failures'):
                failure = response['failures'][0]
                error_message = f"ECS task failed to start for {key}. Reason: {failure.get('reason')}. Detail: {failure.get('detail')}"
                logger.error(error_message)
                # Continue processing other files instead of failing completely
                skipped_files.append({
                    'key': key,
                    'reason': f"ECS task failed: {failure.get('reason')}"
                })
                continue

            if not response.get('tasks'):
                # This case should be rare if failures are handled, but it's good practice
                error_message = f"ECS run_task did not return any tasks or failures for {key}"
                logger.error(error_message)
                skipped_files.append({
                    'key': key,
                    'reason': "ECS task creation returned no tasks"
                })
                continue

            task_arn = response['tasks'][0]['taskArn']
            logger.info(f"Started ECS task for {key}: {task_arn}")
            
            processed_files.append({
                'key': key,
                'task_arn': task_arn,
                'file_size': file_info['size']
            })
        
        # Prepare response
        response_body = {
            'message': f'Processed {len(processed_files)} video files, skipped {len(skipped_files)} files',
            'processed_files': processed_files,
            'skipped_files': skipped_files
        }
        
        # Return success if at least one file was processed, or if no valid video files were found
        status_code = 200 if len(processed_files) > 0 or len(skipped_files) > 0 else 400
        
        logger.info(f"Lambda execution completed: {response_body['message']}")
        
        return {
            'statusCode': status_code,
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Unexpected error: {str(e)}',
                'processed_files': processed_files,
                'skipped_files': skipped_files
            })
        }
