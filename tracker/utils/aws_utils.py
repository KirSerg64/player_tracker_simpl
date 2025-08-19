import boto3
from botocore.client import Config
from loguru import logger
import os

logger = logger.bind(name=__name__)

def load_file_to_bucket(file_path, save_name):
    if not ('AWS_BASE_URL' in os.environ and os.environ['AWS_BASE_URL']):
        logger.error("AWS params are not set")
        return

    svc = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        endpoint_url=os.environ['AWS_BASE_URL'],
        config=Config(s3={'addressing_style': 'virtual'}),
    )

    bucket_name = os.environ['AWS_BUCKET']

    try:
        svc.upload_file(file_path, bucket_name, save_name)
        logger.info(f"Successfully uploaded {file_path} to {bucket_name}")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
