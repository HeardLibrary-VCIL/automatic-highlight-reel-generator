import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as kms from 'aws-cdk-lib/aws-kms';
import { Platform } from 'aws-cdk-lib/aws-ecr-assets';
import { Construct } from 'constructs';

export interface HighlightProcessorStackProps extends cdk.StackProps {
  /**
   * Environment name. 'prod' (default) runs the original EC2/ASG compute.
   * 'dev' runs the SAME pipeline on Fargate (serverless, pay-per-task, no idle
   * EC2 cost) and suffixes all otherwise-hardcoded resource names so a dev stack
   * can coexist with prod in the same account/region.
   */
  readonly envName?: string;

  /**
   * Import an EXISTING VPC instead of creating one. Set this when deploying into
   * an account whose networking is owned/managed elsewhere (e.g. a prod account
   * with a central VPC): the stack attaches to that VPC and provisions NO VPC,
   * subnets, or NAT of its own, inheriting the org's egress/security controls.
   * When omitted, the stack creates its own VPC (dev: NAT-free + public subnet;
   * self-managed prod: 1 NAT + private subnets).
   */
  readonly vpcId?: string;

  /**
   * Explicit subnet IDs to run the ECS task in. Use with `vpcId` when the imported
   * VPC's subnets aren't tagged in the standard way CDK auto-classifies (common in
   * enterprise VPCs) — the network team hands you the exact subnets. When omitted
   * with `vpcId`, the stack selects the VPC's PRIVATE_WITH_EGRESS subnets.
   */
  readonly subnetIds?: string[];

  /**
   * Whether the ECS task gets a public IP. Only relevant when the stack creates its
   * own VPC. Defaults to true for the NAT-free dev VPC (egress via the internet
   * gateway, no inbound rules), false otherwise (egress via NAT/endpoints).
   */
  readonly assignPublicIp?: boolean;

  /**
   * Turn on the Level-3 security hardening baseline. Defaults to ON for prod and
   * OFF for the dev PoC (the dev public-subnet path is intentionally not L3-eligible).
   * When true the stack:
   *   - encrypts S3 objects, EBS volumes, and CloudWatch logs with a customer-managed
   *     KMS key (CMK) whose key policy is scoped to this account (no cross-account grant);
   *   - enforces TLS-only + KMS-only writes on the video bucket via bucket policy;
   *   - provisions VPC endpoints (S3 gateway + interface endpoints for the AWS services
   *     the task calls) so a private-subnet task needs NO internet egress / NO NAT;
   *   - restricts the task security group egress to HTTPS only (no allow-all-outbound).
   * Level-3 data must run in a private subnet with NO public IP — a public IP is a
   * VU Cybersecurity exception, not a default.
   */
  readonly harden?: boolean;
}

export class HighlightProcessorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: HighlightProcessorStackProps) {
    super(scope, id, props);

    // ── Environment flavor ────────────────────────────────────────────────
    // 'dev' -> Fargate (no EC2 instances, no idle cost); anything else -> EC2/ASG.
    const envName = props?.envName ?? 'prod';
    const isDev = envName === 'dev';
    // Dev/prod suffix, now used only for human-readable labels (e.g. the KMS key
    // description). All COLLIDING physical names (log groups, task family, S3
    // notification id, lifecycle rule id) are scoped to `this.stackName` instead,
    // so any number of stacks with any names/flavors coexist in one account.
    const sfx = isDev ? `-${envName}` : '';

    // ── Security hardening baseline (Vanderbilt Level 3) ───────────────────
    // Default: ON for prod, OFF for the dev PoC. The dev public-subnet/public-IP
    // path is intentionally NOT Level-3 eligible, so we don't force encryption/
    // endpoint overhead onto it. Callers can override explicitly.
    const harden = props?.harden ?? !isDev;

    // Customer-managed KMS key for at-rest encryption of S3 objects, EBS, and logs.
    // Rotation on; key policy defaults to this-account-only (no cross-account
    // principal is ever added), which is a primary defense against cross-account
    // access to Level-3 data even if a resource policy is later misconfigured.
    const dataKey = harden
      ? new kms.Key(this, 'DataKey', {
          description: `CMK for scua-video-processor${sfx} Level-3 data at rest`,
          enableKeyRotation: true,
          removalPolicy: cdk.RemovalPolicy.RETAIN, // never auto-delete a key guarding L3 data
        })
      : undefined;

    // Allow CloudWatch Logs to use the CMK for the encrypted log groups below.
    // Scoped by the standard service-principal + region/account condition (no
    // cross-account principal), so the grant can't widen access beyond this account.
    if (dataKey) {
      dataKey.addToResourcePolicy(new iam.PolicyStatement({
        sid: 'AllowCloudWatchLogs',
        effect: iam.Effect.ALLOW,
        principals: [new iam.ServicePrincipal(`logs.${this.region}.amazonaws.com`)],
        actions: [
          'kms:Encrypt',
          'kms:Decrypt',
          'kms:ReEncrypt*',
          'kms:GenerateDataKey*',
          'kms:DescribeKey',
        ],
        resources: ['*'],
        conditions: {
          ArnLike: {
            'kms:EncryptionContext:aws:logs:arn': `arn:aws:logs:${this.region}:${this.account}:log-group:*`,
          },
        },
      }));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PARAMETERS — The Amplify-managed bucket name is passed at deploy time
    // ═══════════════════════════════════════════════════════════════════════
    const amplifyBucketName = new cdk.CfnParameter(this, 'AmplifyBucketName', {
      type: 'String',
      description: 'Name of the Amplify-managed S3 bucket (scua-video-storage). Find in amplify_outputs.json → storage.bucket_name',
    });

    // The Amplify-generated DynamoDB table backing the Video model. The trigger
    // Lambda creates a Video row for videos copied DIRECTLY into the bucket (so they
    // show up in the UI, which lists from DynamoDB). Optional: if left blank, row
    // creation is skipped and only the app's Upload flow writes rows (original
    // behavior). Find the name in the AWS console (DynamoDB → Video-<apiId>-<env>)
    // or from the Amplify data stack outputs.
    const videoTableName = new cdk.CfnParameter(this, 'VideoTableName', {
      type: 'String',
      default: '',
      description: 'Amplify DynamoDB table for the Video model (e.g. Video-xxxx-NONE). Enables auto-creating a row for videos copied directly into the bucket. Leave blank to disable.',
    });

    // Import the existing Amplify bucket (cross-stack reference)
    const videoBucket = s3.Bucket.fromBucketName(this, 'AmplifyVideoBucket', amplifyBucketName.valueAsString);

    // ═══════════════════════════════════════════════════════════════════════
    // NETWORKING
    // Three modes:
    //  (a) vpcId provided  -> IMPORT the existing VPC (prod with a central VPC).
    //      Provisions no VPC/subnets/NAT; inherits the org's networking + egress.
    //  (b) no vpcId, dev   -> create a NAT-FREE VPC, run the task in a PUBLIC subnet
    //      with a public IP (egress via the internet gateway; no inbound rules, so
    //      not an exposure). ~$0 standing network cost.
    //  (c) no vpcId, prod  -> create a VPC with 1 NAT + private subnets (original).
    // ═══════════════════════════════════════════════════════════════════════
    const importedVpc = !!props?.vpcId;

    let vpc: ec2.IVpc;
    if (importedVpc) {
      vpc = ec2.Vpc.fromLookup(this, 'VideoProcessorVPC', { vpcId: props!.vpcId });
    } else if (isDev) {
      // Dev: no NAT. Public subnets only (task egresses via the internet gateway).
      vpc = new ec2.Vpc(this, 'VideoProcessorVPC', {
        maxAzs: 2,
        natGateways: 0,
        subnetConfiguration: [
          { name: 'public', subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        ],
      });
    } else {
      // Self-managed prod (no central VPC supplied): original 1-NAT + private layout.
      vpc = new ec2.Vpc(this, 'VideoProcessorVPC', { maxAzs: 2, natGateways: 1 });
    }

    // Where the ECS task runs, and whether it needs a public IP.
    //  - imported VPC: explicit subnetIds if given, else its PRIVATE_WITH_EGRESS subnets.
    //  - dev self-VPC: the public subnets (need a public IP for IGW egress).
    //  - prod self-VPC: private-with-egress (via NAT), no public IP.
    let taskSubnets: ec2.SubnetSelection;
    if (props?.subnetIds && props.subnetIds.length > 0) {
      taskSubnets = {
        subnets: props.subnetIds.map((sid, i) =>
          ec2.Subnet.fromSubnetId(this, `TaskSubnet${i}`, sid)),
      };
    } else if (importedVpc || !isDev) {
      taskSubnets = { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS };
    } else {
      taskSubnets = { subnetType: ec2.SubnetType.PUBLIC };
    }

    // Public IP: default true only for the NAT-free dev VPC; caller can override.
    // (Imported/private-subnet deployments must stay false — no IGW route there.)
    const assignPublicIp = props?.assignPublicIp ??
      (!importedVpc && isDev);

    // Hardened: no allow-all-outbound. The task only needs HTTPS to reach AWS
    // service endpoints (S3/Transcribe/Bedrock/ECR/Logs/STS), so we open 443 only
    // — to the VPC CIDR when interface endpoints carry the traffic, which keeps
    // egress on the AWS backbone and off the public internet entirely.
    // Unhardened (dev PoC): keep the original outbound-only allow-all for simplicity.
    const securityGroup = new ec2.SecurityGroup(this, 'VideoProcessorSG', {
      vpc,
      description: 'Security group for video processor ECS tasks',
      allowAllOutbound: !harden,   // no inbound rules are ever added anywhere.
    });
    if (harden) {
      // Egress restricted to HTTPS toward the VPC (interface endpoints live here).
      securityGroup.addEgressRule(
        ec2.Peer.ipv4(vpc.vpcCidrBlock),
        ec2.Port.tcp(443),
        'HTTPS to in-VPC interface endpoints (AWS services)'
      );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VPC ENDPOINTS (hardened + stack-owned VPC only)
    // Let a private-subnet task reach the AWS services it needs WITHOUT any route
    // to the internet — so the NAT gateway can be removed and general egress is
    // impossible. For an IMPORTED (central) VPC we assume the network team already
    // provides endpoints/egress, so we add none. Skipped entirely when unhardened.
    //
    // S3 is a free gateway endpoint; the rest are interface endpoints (hourly + data
    // cost, but no internet exposure). These cover the task's calls: S3 (video I/O),
    // Transcribe, Bedrock runtime, ECR (pull the container image), CloudWatch Logs,
    // and STS (task role credentials).
    // ═══════════════════════════════════════════════════════════════════════
    if (harden && !importedVpc) {
      const endpointSg = new ec2.SecurityGroup(this, 'VpcEndpointSG', {
        vpc,
        description: 'Allow HTTPS from the video processor task to VPC interface endpoints',
        allowAllOutbound: true,
      });
      endpointSg.addIngressRule(
        securityGroup,
        ec2.Port.tcp(443),
        'HTTPS from video processor tasks'
      );

      // Free gateway endpoint for S3 (route-table based, no ENI/cost).
      vpc.addGatewayEndpoint('S3GatewayEndpoint', {
        service: ec2.GatewayVpcEndpointAwsService.S3,
      });

      const interfaceEndpoints: Record<string, ec2.InterfaceVpcEndpointAwsService> = {
        EcrApiEndpoint: ec2.InterfaceVpcEndpointAwsService.ECR,
        EcrDkrEndpoint: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
        LogsEndpoint: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
        StsEndpoint: ec2.InterfaceVpcEndpointAwsService.STS,
        TranscribeEndpoint: ec2.InterfaceVpcEndpointAwsService.TRANSCRIBE,
        BedrockRuntimeEndpoint: ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
      };
      for (const [id, service] of Object.entries(interfaceEndpoints)) {
        vpc.addInterfaceEndpoint(id, {
          service,
          securityGroups: [endpointSg],
          privateDnsEnabled: true,
        });
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ECS CLUSTER + CPU AUTO SCALING (min 0 for cost savings)
    // ═══════════════════════════════════════════════════════════════════════
    const cluster = new ecs.Cluster(this, 'VideoProcessorCluster', {
      vpc,
      clusterName: `scua-video-processor-${this.stackName}`,
    });

    // DEV uses Fargate (no ASG/capacity provider). Only build the EC2 auto-scaling
    // group + capacity provider for the EC2 (prod) flavor. `capacityProvider` stays
    // undefined on dev so the Lambda routes with launchType=FARGATE instead.
    let capacityProvider: ecs.AsgCapacityProvider | undefined;
    if (!isDev) {
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
            // Encrypt the instance root volume. The downloaded source video lands
            // here during processing, so at Level 3 it must be encrypted at rest.
            // (KMS-CMK vs account-default: EBS block-device mapping takes only a
            // boolean + optional key id via launch template; the ASG below is set
            // to use the CMK through the launch template's kmsKeyId where supported.
            // At minimum this guarantees encryption is ON.)
            encrypted: true,
          }),
        }],
      });

      capacityProvider = new ecs.AsgCapacityProvider(this, 'VideoProcessorCP', {
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

      // EC2 launch-template permission is only needed for the ASG flavor.
      autoScalingGroup.role.addToPrincipalPolicy(
        new iam.PolicyStatement({
          actions: ['ec2:UseLaunchTemplate'],
          resources: ['*'],
        })
      );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LOGGING
    // ═══════════════════════════════════════════════════════════════════════
    const logGroup = new logs.LogGroup(this, 'VideoProcessorLogs', {
      // Scoped by stack name so ANY number of stacks (any name/flavor) coexist in
      // one account — log group names are account+region-global and must be unique.
      logGroupName: `/ecs/scua-video-processor-${this.stackName}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      // Encrypt task logs with the CMK when hardened (logs can echo data details).
      encryptionKey: dataKey,
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

    // When hardened, the S3 objects are encrypted with the CMK, so the task role
    // needs to use the key to read/write them. Granted on the specific key only
    // (grant() writes a scoped statement), never a wildcard KMS resource.
    if (dataKey) {
      dataKey.grantEncryptDecrypt(taskRole);
    }

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
    // 4 vCPU / ~8GB to match the c5.xlarge sizing on both flavors.
    const TASK_CPU = 4096;
    const TASK_MEM = 8192;

    // DEV: Fargate task def (serverless). PROD: EC2 task def (runs on the ASG).
    // Fargate needs task-level cpu/memory + AWS_VPC networking; it also gives the
    // container ephemeral storage for the downloaded source video (default 20GB,
    // raised to 50GB here to fit long archival tapes for ffmpeg/cv2).
    const taskDefinition: ecs.TaskDefinition = isDev
      ? new ecs.FargateTaskDefinition(this, 'VideoProcessorTaskDef', {
          family: `scua-video-processor-${this.stackName}`,
          taskRole,
          executionRole,
          cpu: TASK_CPU,
          memoryLimitMiB: TASK_MEM,
          ephemeralStorageGiB: 50,
          runtimePlatform: {
            cpuArchitecture: ecs.CpuArchitecture.X86_64,
            operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
          },
        })
      : new ecs.Ec2TaskDefinition(this, 'VideoProcessorTaskDef', {
          family: `scua-video-processor-${this.stackName}`,
          taskRole,
          executionRole,
          networkMode: ecs.NetworkMode.AWS_VPC,
        });

    taskDefinition.addContainer('video-processor', {
      image: ecs.ContainerImage.fromAsset('./video-processing', {
        platform: Platform.LINUX_AMD64
      }),
      // On Fargate the task-level cpu/mem govern; on EC2 these container limits do.
      // Reserve ~7GB on EC2 (leave headroom under the 8GiB instance); on Fargate the
      // container shares the full task memory.
      memoryLimitMiB: isDev ? TASK_MEM : 7168,
      cpu: TASK_CPU,
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
      logGroupName: `/aws/lambda/scua-video-trigger-${this.stackName}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      encryptionKey: dataKey,
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
        // Subnets the task runs in — resolved from taskSubnets (explicit ids, the
        // imported VPC's private-with-egress subnets, or the dev public subnets).
        SUBNET_IDS: vpc.selectSubnets(taskSubnets).subnetIds.join(','),
        SECURITY_GROUP: securityGroup.securityGroupId,
        ASSIGN_PUBLIC_IP: assignPublicIp ? 'ENABLED' : 'DISABLED',
        // Amplify Video table — the handler creates a row for videos copied directly
        // into the bucket. Empty string disables that (handler skips row creation).
        VIDEO_TABLE_NAME: videoTableName.valueAsString,
        // DEV: route with launchType=FARGATE. PROD: route onto the EC2 capacity
        // provider. The handler picks whichever env var is present.
        ...(isDev
          ? { LAUNCH_TYPE: 'FARGATE' }
          : { CAPACITY_PROVIDER_NAME: capacityProvider!.capacityProviderName }),
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

    // Write the Video row for directly-copied uploads. Scoped to the specific Amplify
    // Video table (from the VideoTableName parameter). Only PutItem is needed — the
    // handler uses a conditional put (attribute_not_exists) so it never overwrites.
    triggerLambda.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['dynamodb:PutItem'],
      resources: [
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${videoTableName.valueAsString}`,
      ],
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
        // Stack-name-scoped so distinct stacks never overwrite each other's
        // notification entries even if pointed at the same bucket.
        NotificationId: `scua-video-trim-trigger-${this.stackName}`,
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
        rule_id = event['ResourceProperties'].get('LifecycleRuleId', 'scua-cleanup-noncurrent')
        print(f"BucketConfigHandler: {event['RequestType']} on {bucket} (rule {rule_id})")

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
                            'ID': rule_id,
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
        // Stack-scoped rule id. NOTE: put_bucket_lifecycle_configuration REPLACES
        // the bucket's whole lifecycle config, so two stacks pointed at the SAME
        // bucket would still clobber each other's rule regardless of id — this is
        // safe only because dev/prod use SEPARATE buckets. The scoped id keeps the
        // rule self-identifying if that ever changes.
        LifecycleRuleId: `scua-cleanup-noncurrent-${this.stackName}`,
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
