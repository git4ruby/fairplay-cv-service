"""
Configuration settings for FairPlay CV Service
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # AWS S3 Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    aws_s3_bucket: str

    # Service Configuration
    service_port: int = 8000
    service_host: str = "0.0.0.0"
    debug: bool = True

    # Processing Configuration
    max_video_size_mb: int = 50
    processing_timeout_seconds: int = 30
    confidence_threshold: float = 0.6

    # Shuttle Detection Settings
    shuttle_min_confidence: float = 0.5
    frame_skip: int = 2

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
