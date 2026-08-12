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

        # Get environment variables
        cluster = os.environ['CLUSTER_NAME']
        task_definition = os.environ['TASK_DEFINITION']
        subnet_ids = os.environ['SUBNET_IDS'].split(',')
        security_group = os.environ['SECURITY_GROUP']
        assign_public_ip = os.environ['ASSIGN_PUBLIC_IP']
        capacity_provider_name = os.environ['CAPACITY_PROVIDER_NAME']

        # Start ECS task in TRIM mode
        response = ecs.run_task(
            cluster=cluster,
            capacityProviderStrategy=[
                {'capacityProvider': capacity_provider_name, 'weight': 1},
            ],
            taskDefinition=task_definition,
            count=1,
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnet_ids,
                    'assignPublicIp': assign_public_ip,
                    'securityGroups': [security_group]
                }
            },
            overrides={
                'containerOverrides': [{
                    'name': 'video-processor',
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': bucket},
                        {'name': 'S3_KEY', 'value': video_key},
                        {'name': 'TRIM_REQUEST_KEY', 'value': trim_request_key},
                        {'name': 'MODE', 'value': 'trim'},
                    ]
                }]
            }
        )

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
    """Launch an ECS task in DETECT mode to RE-RUN AI segmentation on a video.
    Overwrites segment/<basename>.json with fresh detector output; it does NOT
    create any new video — only the segments change."""
    try:
        response = s3.get_object(Bucket=bucket, Key=segment_request_key)
        req = json.loads(response['Body'].read().decode('utf-8'))
        video_key = req.get('video_key', '')

        if not video_key:
            logger.error(f"Segment request missing video_key: {segment_request_key}")
            return {'key': segment_request_key, 'reason': 'Missing video_key'}

        cluster = os.environ['CLUSTER_NAME']
        task_definition = os.environ['TASK_DEFINITION']
        subnet_ids = os.environ['SUBNET_IDS'].split(',')
        security_group = os.environ['SECURITY_GROUP']
        assign_public_ip = os.environ['ASSIGN_PUBLIC_IP']
        capacity_provider_name = os.environ['CAPACITY_PROVIDER_NAME']

        response = ecs.run_task(
            cluster=cluster,
            capacityProviderStrategy=[
                {'capacityProvider': capacity_provider_name, 'weight': 1},
            ],
            taskDefinition=task_definition,
            count=1,
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnet_ids,
                    'assignPublicIp': assign_public_ip,
                    'securityGroups': [security_group]
                }
            },
            overrides={
                'containerOverrides': [{
                    'name': 'video-processor',
                    'environment': [
                        {'name': 'S3_BUCKET', 'value': bucket},
                        {'name': 'S3_KEY', 'value': video_key},
                        {'name': 'MODE', 'value': 'detect'},
                    ]
                }]
            }
        )

        if response.get('failures'):
            failure = response['failures'][0]
            logger.error(f"ECS segment task failed: {failure.get('reason')}")
            return {'key': segment_request_key, 'reason': f"ECS failed: {failure.get('reason')}"}

        task_arn = response['tasks'][0]['taskArn']
        logger.info(f"Started ECS segment (re-run) task: {task_arn}")
        return {'key': segment_request_key, 'task_arn': task_arn, 'mode': 'detect(resegment)'}

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
            
            logger.info(f"Processing file: s3://{bucket}/{key}")
            
            # Route: edit/*_trim_request.json → trim mode
            if key.startswith('edit/') and key.endswith('_trim_request.json'):
                logger.info(f"Trim request detected: {key}")
                processed_files.append(launch_trim_task(bucket, key))
                continue

            # Route: edit/*_segment_request.json → re-run AI segmentation (detect mode)
            if key.startswith('edit/') and key.endswith('_segment_request.json'):
                logger.info(f"Segment re-run request detected: {key}")
                processed_files.append(launch_segment_task(bucket, key))
                continue
            
            # Route: video/*.mp4 → segment detection mode. Anything else under edit/
            # (e.g. the rendered .mp4 outputs) is not a request we act on — skip it.
            if not key.startswith('video/'):
                logger.info(f"Skipping {key}: not a video/ upload or an edit/ trim/segment request")
                skipped_files.append({'key': key, 'reason': 'Not a video/ upload or an edit/ trim/segment request'})
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

            # Get environment variables
            cluster = os.environ['CLUSTER_NAME']
            task_definition = os.environ['TASK_DEFINITION']
            subnet_ids = os.environ['SUBNET_IDS'].split(',')
            security_group = os.environ['SECURITY_GROUP']
            assign_public_ip = os.environ['ASSIGN_PUBLIC_IP']
            capacity_provider_name = os.environ['CAPACITY_PROVIDER_NAME']

            # Start ECS task
            response = ecs.run_task(
                cluster=cluster,
                capacityProviderStrategy=[
                    {
                        'capacityProvider': capacity_provider_name,
                        'weight': 1,
                    },
                ],
                taskDefinition=task_definition,
                count=1,
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': subnet_ids,
                        'assignPublicIp': assign_public_ip,
                        'securityGroups': [security_group]
                    }
                },
                overrides={
                    'containerOverrides': [
                        {
                            'name': 'video-processor',
                            'environment': [
                                {
                                    'name': 'S3_BUCKET',
                                    'value': bucket
                                },
                                {
                                    'name': 'S3_KEY',
                                    'value': key
                                }
                            ]
                        }
                    ]
                }
            )
            
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
