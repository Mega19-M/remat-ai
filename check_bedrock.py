"""Check AWS credentials and Amazon Bedrock access, and list model ids you can try.

Usage:  python scripts/check_bedrock.py [model-id-to-test]
Needs IAM permissions: sts:GetCallerIdentity, bedrock:ListFoundationModels, bedrock:ListInferenceProfiles,
bedrock:InvokeModel (for the optional test call).
"""
import os
import sys

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    sys.exit("boto3 is not installed. Run: pip install -r requirements.txt")

region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
print("Region:", region)
try:
    who = boto3.client("sts", region_name=region).get_caller_identity()
    print("Signed in as:", who["Arn"])
except (BotoCoreError, ClientError) as exc:
    sys.exit(f"Could not authenticate with AWS: {exc}\nRun `aws configure` first (see README step 6).")

bedrock = boto3.client("bedrock", region_name=region)
try:
    models = bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
    print("\nText models visible in this region (visible does not always mean access is granted):")
    for m in models:
        print("  ", m["modelId"], "-", m.get("providerName", ""))
except (BotoCoreError, ClientError) as exc:
    print("Could not list foundation models:", exc)
try:
    profiles = bedrock.list_inference_profiles()["inferenceProfileSummaries"]
    print("\nInference profiles (some models must be called through these ids):")
    for p in profiles:
        print("  ", p["inferenceProfileId"])
except (BotoCoreError, ClientError) as exc:
    print("Could not list inference profiles:", exc)

model_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REMAT_BEDROCK_MODEL_ID", "")
if model_id:
    print("\nTesting model:", model_id)
    try:
        out = boto3.client("bedrock-runtime", region_name=region).converse(
            modelId=model_id, messages=[{"role": "user", "content": [{"text": "Reply with the single word: ready"}]}],
            inferenceConfig={"maxTokens": 20})
        print("Model replied:", out["output"]["message"]["content"][0]["text"])
    except (BotoCoreError, ClientError) as exc:
        sys.exit(f"Model test failed: {exc}\nCheck that the model is enabled for your account and the id is right.")
else:
    print("\nPass a model id to test it:  python scripts/check_bedrock.py <model-id>")
