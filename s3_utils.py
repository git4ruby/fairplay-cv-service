"""
AWS S3 utilities for video and frame storage
"""
import boto3
import logging
from pathlib import Path
from typing import Optional
from botocore.exceptions import ClientError
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class S3Handler:
    """Handles S3 operations for video and frame storage"""

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket = settings.aws_s3_bucket

    def download_video(self, video_url: str, local_path: str) -> bool:
        """
        Download video from S3 to local path

        Args:
            video_url: S3 URL or signed URL
            local_path: Local file path to save video

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract S3 key from URL
            # Handle both formats:
            # - https://s3.amazonaws.com/bucket/key
            # - https://bucket.s3.amazonaws.com/key
            # - https://bucket.s3.region.amazonaws.com/key
            # - Signed URLs with query params

            # Remove query parameters if present (signed URLs)
            base_url = video_url.split("?")[0]

            # Extract key from URL
            if f"{self.bucket}.s3" in base_url:
                # Format: https://bucket.s3.region.amazonaws.com/key
                s3_key = base_url.split(f"{self.bucket}.s3")[1].split("/", 2)[-1]
            elif "s3.amazonaws.com" in base_url:
                # Format: https://s3.amazonaws.com/bucket/key
                parts = base_url.split("s3.amazonaws.com/")[1].split("/", 1)
                if len(parts) > 1:
                    s3_key = parts[1]
                else:
                    s3_key = parts[0]
            else:
                # Assume it's just the key
                s3_key = base_url.split("/")[-1]

            logger.info(f"Downloading video from S3: {s3_key}")

            # Download file
            self.s3_client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=local_path
            )

            logger.info(f"Video downloaded successfully to {local_path}")
            return True

        except ClientError as e:
            logger.error(f"S3 download error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading video: {e}")
            return False

    def upload_frame(self, local_path: str, s3_key: Optional[str] = None) -> Optional[str]:
        """
        Upload landing frame image to S3

        Args:
            local_path: Local path to the frame image
            s3_key: S3 key to use (if None, will use filename)

        Returns:
            S3 URL of uploaded frame, or None if failed
        """
        try:
            if s3_key is None:
                s3_key = f"landing_frames/{Path(local_path).name}"

            logger.info(f"Uploading frame to S3: {s3_key}")

            # Upload file
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket,
                Key=s3_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )

            # Generate URL
            frame_url = f"https://{self.bucket}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"

            logger.info(f"Frame uploaded successfully: {frame_url}")
            return frame_url

        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading frame: {e}")
            return None

    def generate_signed_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for private S3 object

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default 1 hour)

        Returns:
            Signed URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating signed URL: {e}")
            return None
