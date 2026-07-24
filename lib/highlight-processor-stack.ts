import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
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
    // ECS CLUSTER + GPU AUTO SCALING (min 0 for cost savings)
    // ═══════════════════════════════════════════════════════════════════════
    const cluster = new ecs.Cluster(this, 'VideoProcessorCluster', {
      vpc,
      clusterName: `scua-video-processor-${this.stackName}`,
    });

    const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'VideoProcessorASG', {
      vpc,
      // CPU instance for dead-space detection (ffmpeg-only, no GPU needed)
      // Switch back to G4DN when GPU quota is approved for VLM content search
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
      enableManagedTerminationProtection: false,
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
      ],
    }));

    // ── Amazon Transcribe (Stage A of the segmentation pipeline) ──────────────
    // Transcribe reads the media straight out of the Amplify bucket using THIS
    // role's S3 permissions (same-account access), so the video never has to be
    // copied anywhere. With no OutputBucketName the result lands in a
    // service-managed bucket and comes back as a presigned URL, so no extra
    // write permission is needed. Job ARNs are minted per run -> resource '*'.
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'transcribe:StartTranscriptionJob',
        'transcribe:GetTranscriptionJob',
      ],
      resources: ['*'],
    }));

    // ── Amazon Bedrock (Stages C/D: taxonomy discovery + multimodal labeling) ─
    // Both the frame classification and the transcript labeling call Claude via
    // Bedrock, so there is no ANTHROPIC_API_KEY anywhere in the stack. Invoking a
    // cross-region inference profile (us.anthropic.*) requires permission on BOTH
    // the profile ARN and the foundation models it routes to.
    taskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['bedrock:InvokeModel'],
      resources: [
        'arn:aws:bedrock:*::foundation-model/anthropic.*',
        `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/*`,
      ],
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
        platform: Platform.LINUX_AMD64,
        buildArgs: {
          'HUGGINGFACE_TOKEN': process.env.HUGGINGFACE_TOKEN || ''
        }
      }),
      memoryLimitMiB: 7168,  // ~7GB for c5.xlarge (8GiB total)
      cpu: 4096,             // 4 vCPUs for c5.xlarge
      // gpuCount: 1,        // Re-enable when switching back to G4DN
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'video-processor',
        logGroup,
      }),
      command: ["python3", "main.py"],
      environment: {
        // botocore resolves region from AWS_DEFAULT_REGION; set both so every
        // boto3 client (Transcribe has no global-endpoint fallback) has a region.
        AWS_REGION: this.region,
        AWS_DEFAULT_REGION: this.region,
        // Output paths matching SCUA frontend storage conventions
        RESULT_PREFIX: 'edit',
        SEGMENT_PREFIX: 'segment',
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

            # Remove any existing notifications with our IDs
            lambda_configs = existing.get('LambdaFunctionConfigurations', [])
            our_ids = {notification_id, notification_id + '-trim'}
            lambda_configs = [c for c in lambda_configs if c.get('Id') not in our_ids]

            # Add our notifications
            lambda_configs.append({
                'Id': notification_id,
                'LambdaFunctionArn': lambda_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'video/'}
                        ]
                    }
                }
            })
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
            existing['LambdaFunctionConfigurations'] = lambda_configs
            s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=existing)

        elif event['RequestType'] == 'Delete':
            existing = s3.get_bucket_notification_configuration(Bucket=bucket)
            existing.pop('ResponseMetadata', None)
            lambda_configs = existing.get('LambdaFunctionConfigurations', [])
            lambda_configs = [c for c in lambda_configs if c.get('Id') not in (notification_id, notification_id + '-trim')]
            existing['LambdaFunctionConfigurations'] = lambda_configs
            s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=existing)

        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as e:
        print(f"Error: {e}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)})
`),
    });

    notificationHandler.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['s3:GetBucketNotification', 's3:PutBucketNotification'],
      resources: [videoBucket.bucketArn],
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
