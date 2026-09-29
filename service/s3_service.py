import os
import boto3
from botocore.exceptions import ClientError

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
    return _s3_client

def upload_to_s3(user_id: str, prefix: str, file_id: str, file_content: bytes, content_type: str = "application/json"):
    """
    Uploads a file to S3, namespaced by user_id.
    e.g. upload_to_s3("user_123", "templates", "tpl_456.json", b"...", "application/json")
    -> Key: "user_123/templates/tpl_456.json"
    """
    client = get_s3_client()
    if not client:
        print("[S3] AWS credentials not configured. Skipping upload.")
        return False
        
    s3_key = f"{user_id}/{prefix}/{file_id}"
    try:
        client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        return True
    except ClientError as e:
        print(f"[S3] Upload failed: {e}")
        return False

def download_from_s3(user_id: str, prefix: str, file_id: str) -> bytes:
    """
    Downloads a file from S3, namespaced by user_id.
    """
    client = get_s3_client()
    if not client:
        return b""
        
    s3_key = f"{user_id}/{prefix}/{file_id}"
    try:
        response = client.get_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
        return response["Body"].read()
    except ClientError as e:
        print(f"[S3] Download failed: {e}")
        return b""

def delete_from_s3(user_id: str, prefix: str, file_id: str):
    """
    Deletes a file from S3.
    """
    client = get_s3_client()
    if not client:
        return False
        
    s3_key = f"{user_id}/{prefix}/{file_id}"
    try:
        client.delete_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
        return True
    except ClientError as e:
        print(f"[S3] Delete failed: {e}")
        return False

def list_from_s3(user_id: str, prefix: str) -> list:
    """
    Lists files in a specific user's prefix directory.
    Returns a list of keys.
    """
    client = get_s3_client()
    if not client:
        return []
        
    s3_prefix = f"{user_id}/{prefix}/"
    try:
        response = client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=s3_prefix)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"]]
        return []
    except ClientError as e:
        print(f"[S3] List failed: {e}")
        return []
