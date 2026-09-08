import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import { Platform } from 'aws-cdk-lib/aws-ecr-assets';
import { Construct } from 'constructs';

export class HighlightProcessorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ═══════════════════════════════════════════════════════════════════════
    // PARAMETERS — The Amplify-managed bucket name is passed at deploy time
    // ═══════════════════════════════════════════════════════════════════════
    const amplifyBucketName = new cdk.CfnParameter(this, 'AmplifyBucketName', {
      type: 'String',
      description: 'Name of the Amplify-managed S3 bucket (scua-video-storage). Find in amplify_outputs.json → storage.bucket_name',
    });

    // Import the existing Amplify bucket (cross-stack reference)
    const videoBucket = s3.Bucket.fromBucketName(this, 'AmplifyVideoBucket', amplifyBucketName.valueAsString);

    // ═══════════════════════════════════════════════════════════════════════
    // NETWORKING
    // ═══════════════════════════════════════════════════════════════════════
    const vpc = new ec2.Vpc(this, 'VideoProcessorVPC', {
      maxAzs: 2,
      natGateways: 1,
    });

    const securityGroup = new ec2.SecurityGroup(this, 'VideoProcessorSG', {
      vpc,
      description: 'Security group for video processor ECS tasks',
      allowAllOutbound: true,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // ECS CLUSTER + CPU AUTO SCALING (min 0 for cost savings)
    // ═══════════════════════════════════════════════════════════════════════
    const cluster = new ecs.Cluster(this, 'VideoProcessorCluster', {
      vpc,
      clusterName: `scua-video-processor-${this.stackName}`,
    });

    const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'VideoProcessorASG', {
      vpc,

      instanceType: ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.XLARGE),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2023(),
      minCapacity: 0,
      maxCapacity: 2,
      securityGroup,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: autoscaling.BlockDeviceVolume.ebs(100, {
          deleteOnTermination: true,
          volumeType: autoscaling.EbsDeviceVolumeType.GP3,
        }),
      }],
    });

    const capacityProvider = new ecs.AsgCapacityProvider(this, 'VideoProcessorCP', {
      autoScalingGroup,
      enableManagedScaling: true,
      // Protect instances that have a running task from scale-in. Video
      // processing is a long batch job (minutes); without this, the ASG can
      // terminate the instance mid-run, killing the task before it finishes
      // (segmentation silently never completes). ECS scales the instance down
      // only after its task ends.
      enableManagedTerminationProtection: true,
    });

    cluster.addAsgCapacityProvider(capacityProvider);

    // ═══════════════════════════════════════════════════════════════════════
    // LOGGING
    // ═══════════════════════════════════════════════════════════════════════
    const logGroup = new logs.LogGroup(this, 'VideoProcessorLogs', {
      logGroupName: `/ecs/scua-video-processor`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // IAM ROLES
    // ═══════════════════════════════════════════════════════════════════════
    const taskRole = new iam.Role(this, 'VideoProcessorTaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    const executionRole = new iam.Role(this, 'VideoProcessorExecutionRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    autoScalingGroup.role.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ['ec2:UseLaunchTemplate'],
        resources: ['*'],
      })
    );

    // Grant ECS task role access to the Amplify bucket
    // Read from video/*, write to edit/* and segment/*
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['s3:GetObject', 's3:HeadObject'],
      resources: [videoBucket.arnForObjects('video/*')],
    }));
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['s3:PutObject', 's3:GetObject'],
      resources: [
        videoBucket.arnForObjects('edit/*'),
        videoBucket.arnForObjects('segment/*'),
        videoBucket.arnForObjects('review/*'),
        videoBucket.arnForObjects('transcript/*'),
      ],
    }));

    // ── Amazon Transcribe (full-video speech-to-text with speaker diarization) ─
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'transcribe:StartTranscriptionJob',
        'transcribe:GetTranscriptionJob',
      ],
      resources: ['*'],
    }));

    // ── Amazon Bedrock (Claude via cross-region inference profile) ─────────────
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['bedrock:InvokeModel'],
      resources: [
        'arn:aws:bedrock:*::foundation-model/anthropic.*',
        `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/*`,
      ],
    }));

    // Marketplace permissions required for newer Bedrock models (Sonnet 4+)
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'aws-marketplace:ViewSubscriptions',
        'aws-marketplace:Subscribe',
      ],
      resources: ['*'],
    }));

    // ═══════════════════════════════════════════════════════════════════════
    // ECS TASK DEFINITION
    // ═══════════════════════════════════════════════════════════════════════
    const taskDefinition = new ecs.Ec2TaskDefinition(this, 'VideoProcessorTaskDef', {
      family: 'scua-video-processor',
      taskRole,
      executionRole,
      networkMode: ecs.NetworkMode.AWS_VPC,
    });

    taskDefinition.addContainer('video-processor', {
      image: ecs.ContainerImage.fromAsset('./video-processing', {
        platform: Platform.LINUX_AMD64
      }),
      memoryLimitMiB: 7168,  // ~7GB for c5.xlarge (8GiB total)
      cpu: 4096,             // 4 vCPUs for c5.xlarge
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'video-processor',
        logGroup,
      }),
      command: ["python3", "main.py"],
      environment: {
        AWS_REGION: this.region,
        AWS_DEFAULT_REGION: this.region,
        // Output paths matching SCUA frontend storage conventions
        RESULT_PREFIX: 'edit',
        SEGMENT_PREFIX: 'segment',
        // Content labeling: "auto" = always on (Bedrock via task role); "off" = dead-space only
        CONTENT_SEGMENT: 'auto',
      },
      essential: true,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // TRIGGER LAMBDA — fired by S3 video/* uploads on the Amplify bucket
    // ═══════════════════════════════════════════════════════════════════════
    const triggerLambdaLogGroup = new logs.LogGroup(this, 'TriggerLambdaLogGroup', {
      logGroupName: `/aws/lambda/scua-video-trigger`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const triggerLambda = new lambda.Function(this, 'VideoTriggerLambda', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handler.lambda_handler',
      code: lambda.Code.fromAsset('./lambda'),
      logGroup: triggerLambdaLogGroup,
      timeout: cdk.Duration.minutes(5),
      environment: {
        CLUSTER_NAME: cluster.clusterName,
        TASK_DEFINITION: taskDefinition.taskDefinitionArn,
        SUBNET_IDS: vpc.privateSubnets.map(subnet => subnet.subnetId).join(','),
        SECURITY_GROUP: securityGroup.securityGroupId,
        ASSIGN_PUBLIC_IP: 'DISABLED',
        CAPACITY_PROVIDER_NAME: capacityProvider.capacityProviderName,
      },
    });

    // Lambda permissions
    triggerLambda.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['ecs:RunTask'],
      resources: [taskDefinition.taskDefinitionArn],
    }));

    triggerLambda.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['s3:GetObject', 's3:GetObjectAttributes', 's3:HeadObject'],
      resources: [videoBucket.arnForObjects('video/*'), videoBucket.arnForObjects('edit/*')],
    }));

    triggerLambda.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['iam:PassRole'],
      resources: [taskRole.roleArn, executionRole.roleArn],
    }));

    // ═══════════════════════════════════════════════════════════════════════
    // S3 NOTIFICATION — trigger Lambda on video/* uploads
    // Since this is an imported bucket, we must add the notification manually
    // via a custom resource or use bucket notification configuration.
    // ═══════════════════════════════════════════════════════════════════════

    // Allow S3 to invoke the Lambda
    triggerLambda.addPermission('AllowS3Invoke', {
      principal: new iam.ServicePrincipal('s3.amazonaws.com'),
      sourceArn: videoBucket.bucketArn,
      sourceAccount: this.account,
    });

    // Custom resource to add notification to the existing Amplify bucket
    const notificationHandler = new lambda.Function(this, 'S3NotificationHandler', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      timeout: cdk.Duration.minutes(2),
      code: lambda.Code.fromInline(`
import boto3
import cfnresponse
import json

# Stable physical id so CloudFormation does IN-PLACE updates (never a replacement,
# whose delete-of-the-old-resource would wipe the config the new one just wrote).
STABLE_ID = 'scua-s3-notification-config'

def handler(event, context):
    try:
        s3 = boto3.client('s3')
        bucket = event['ResourceProperties']['BucketName']
        lambda_arn = event['ResourceProperties']['LambdaArn']
        notification_id = event['ResourceProperties'].get('NotificationId', 'scua-video-trigger')

        if event['RequestType'] in ['Create', 'Update']:
            # Get existing notification config
            existing = s3.get_bucket_notification_configuration(Bucket=bucket)
            existing.pop('ResponseMetadata', None)

            # Remove any config that targets OUR Lambda (any id — covers legacy ids
            # like scua-trim-request-trigger) so we never leave an overlapping edit/
            # rule behind, plus our known ids, before re-adding a clean set.
            lambda_configs = existing.get('LambdaFunctionConfigurations', [])
            our_ids = {notification_id, notification_id + '-trim', notification_id + '-segment'}
            lambda_configs = [c for c in lambda_configs
                              if c.get('LambdaFunctionArn') != lambda_arn and c.get('Id') not in our_ids]

            # video/ uploads -> segment detection.
            # Real uploads only, NOT ObjectCreated:Copy. The editor renames a video
            # by copying it to the new key (SCUA-Video-Editing src/utils/rename.ts),
            # and on 'Copy' this rule would re-run the whole pipeline over the
            # renamed object -- burning Bedrock quota and overwriting the segment
            # JSON the rename just moved, destroying the user's manual edits.
            lambda_configs.append({
                'Id': notification_id,
                'LambdaFunctionArn': lambda_arn,
                'Events': ['s3:ObjectCreated:Put',
                           's3:ObjectCreated:Post',
                           's3:ObjectCreated:CompleteMultipartUpload'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'video/'}
                        ]
                    }
                }
            })

            # edit/*_trim_request.json -> trim mode
            lambda_configs.append({
                'Id': notification_id + '-trim',
                'LambdaFunctionArn': lambda_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'edit/'},
                            {'Name': 'suffix', 'Value': '_trim_request.json'}
                        ]
                    }
                }
            })

            # edit/*_segment_request.json -> re-segmentation
            lambda_configs.append({
                'Id': notification_id + '-segment',
                'LambdaFunctionArn': lambda_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'edit/'},
                            {'Name': 'suffix', 'Value': '_segment_request.json'}
                        ]
                    }
                }
            })
            existing['LambdaFunctionConfigurations'] = lambda_configs
            s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=existing)

        elif event['RequestType'] == 'Delete':
            # Best-effort cleanup: removing our notification entries is not worth
            # failing (and rolling back) the whole stack. This DELETE also runs when
            # the BucketName parameter CHANGES (CloudFormation replaces the resource
            # and deletes the OLD one) — and the old bucket may be a DIFFERENT Amplify
            # branch bucket the handler's role can't touch, or may be gone entirely.
            # Swallow any error and still report SUCCESS so the deploy proceeds.
            try:
                existing = s3.get_bucket_notification_configuration(Bucket=bucket)
                existing.pop('ResponseMetadata', None)
                lambda_configs = existing.get('LambdaFunctionConfigurations', [])
                lambda_configs = [c for c in lambda_configs if c.get('Id') not in (notification_id, notification_id + '-trim', notification_id + '-segment')]
                existing['LambdaFunctionConfigurations'] = lambda_configs
                s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=existing)
            except Exception as del_err:
                print(f"Delete cleanup skipped (non-fatal) for bucket {bucket}: {del_err}")

        cfnresponse.send(event, context, cfnresponse.SUCCESS, {}, STABLE_ID)
    except Exception as e:
        print(f"Error: {e}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)}, STABLE_ID)
`),
    });

    // The notification handler must manage the CURRENT bucket, but on a BucketName
    // parameter change CloudFormation replaces this resource and runs a DELETE against
    // the PREVIOUS bucket (often a sibling Amplify branch bucket). Grant the notif
    // actions on any Amplify storage bucket in this account/region so that cross-branch
    // cleanup DELETE can succeed instead of AccessDenied. (Scoped to Amplify buckets,
    // not all of S3.)
    notificationHandler.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['s3:GetBucketNotification', 's3:PutBucketNotification'],
      resources: [
        videoBucket.bucketArn,
        'arn:aws:s3:::amplify-*-scuavideostoragebucket*',
      ],
    }));

    new cdk.CustomResource(this, 'S3NotificationConfig', {
      serviceToken: notificationHandler.functionArn,
      properties: {
        BucketName: amplifyBucketName.valueAsString,
        LambdaArn: triggerLambda.functionArn,
        NotificationId: 'scua-video-trim-trigger',
      },
    });

    // ═══════════════════════════════════════════════════════════════════════
    // BUCKET CONFIG — disable versioning + 7-day lifecycle for noncurrent versions
    // The Amplify bucket has versioning enabled by default. We suspend it and
    // add a lifecycle rule to expire noncurrent versions after 1 day (cleanup).
    // ═══════════════════════════════════════════════════════════════════════
    const bucketConfigHandler = new lambda.Function(this, 'BucketConfigHandler', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      timeout: cdk.Duration.minutes(2),
      code: lambda.Code.fromInline(`
import boto3
import cfnresponse
import traceback

def handler(event, context):
    try:
        s3 = boto3.client('s3')
        bucket = event['ResourceProperties']['BucketName']
        print(f"BucketConfigHandler: {event['RequestType']} on {bucket}")

        if event['RequestType'] in ['Create', 'Update']:
            # Suspend versioning
            try:
                s3.put_bucket_versioning(
                    Bucket=bucket,
                    VersioningConfiguration={'Status': 'Suspended'}
                )
                print(f"Versioning suspended on {bucket}")
            except Exception as e:
                print(f"Warning: could not suspend versioning: {e}")
                # Non-fatal — continue to lifecycle

            # Add lifecycle rule: expire noncurrent versions after 1 day,
            # delete expired delete markers, and abort incomplete multipart after 7 days
            try:
                s3.put_bucket_lifecycle_configuration(
                    Bucket=bucket,
                    LifecycleConfiguration={
                        'Rules': [{
                            'ID': 'scua-cleanup-noncurrent',
                            'Status': 'Enabled',
                            'Filter': {'Prefix': ''},
                            'NoncurrentVersionExpiration': {'NoncurrentDays': 1},
                            'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7},
                        }]
                    }
                )
                print(f"Lifecycle rule set on {bucket}")
            except Exception as e:
                print(f"Warning: could not set lifecycle: {e}")
                # Non-fatal

        # On Delete: leave bucket as-is (don't re-enable versioning)
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)})
`),
    });

    bucketConfigHandler.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:PutBucketVersioning',
        's3:GetBucketVersioning',
        's3:PutLifecycleConfiguration',
        's3:GetLifecycleConfiguration',
        's3:PutBucketLifecycleConfiguration',
        's3:GetBucketLifecycleConfiguration',
      ],
      resources: [videoBucket.bucketArn],
    }));

    new cdk.CustomResource(this, 'BucketVersioningConfig', {
      serviceToken: bucketConfigHandler.functionArn,
      properties: {
        BucketName: amplifyBucketName.valueAsString,
      },
    });

    // ═══════════════════════════════════════════════════════════════════════
    // OUTPUTS
    // ═══════════════════════════════════════════════════════════════════════
    new cdk.CfnOutput(this, 'AmplifyBucket', {
      value: amplifyBucketName.valueAsString,
      description: 'Amplify-managed S3 bucket being used for video storage',
    });

    new cdk.CfnOutput(this, 'ClusterName', {
      value: cluster.clusterName,
      description: 'ECS cluster name',
    });

    new cdk.CfnOutput(this, 'LogGroupName', {
      value: logGroup.logGroupName,
      description: 'CloudWatch log group for ECS tasks',
    });

    new cdk.CfnOutput(this, 'TaskDefinitionArn', {
      value: taskDefinition.taskDefinitionArn,
      description: 'Task definition ARN',
    });
  }
}
