#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { HighlightProcessorStack } from '../lib/highlight-processor-stack';

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

// Optional VPC injection via context (so networking isn't hardcoded):
//   -c vpcId=vpc-xxxx            import an existing VPC (prod with a central VPC)
//   -c subnetIds=subnet-a,subnet-b   run the task in these exact subnets
// When vpcId is omitted, each stack creates its own VPC (dev: NAT-free/public;
// self-managed prod: 1 NAT + private).
const vpcId = app.node.tryGetContext('vpcId') as string | undefined;
const subnetIdsCtx = app.node.tryGetContext('subnetIds') as string | undefined;
const subnetIds = subnetIdsCtx ? subnetIdsCtx.split(',').map(s => s.trim()).filter(Boolean) : undefined;

// PROD: original EC2/ASG compute. Attaches to a provided VPC when `vpcId` context
// is set (prod account with a central VPC); otherwise creates its own (unchanged).
//   npx cdk deploy HighlightProcessorStack \
//     --parameters AmplifyBucketName=<prod bucket> \
//     -c vpcId=<prod-vpc-id> [-c subnetIds=<id1,id2>]
new HighlightProcessorStack(app, 'HighlightProcessorStack', {
  env,
  vpcId,
  subnetIds,
});

// DEV: same pipeline on Fargate (serverless, no idle EC2 cost), wired to the
// Amplify DEV bucket. No VPC/NAT of its own beyond a NAT-free public-subnet VPC.
//   npx cdk deploy HighlightProcessorDevStack \
//     --parameters AmplifyBucketName=<dev bucket from the dev amplify_outputs.json>
new HighlightProcessorStack(app, 'HighlightProcessorDevStack', {
  env,
  envName: 'dev',
});