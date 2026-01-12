"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal


class AnalyzeRequest(BaseModel):
    """Request schema for video analysis"""

    video_url: str = Field(
        ...,
        description="S3 URL or signed URL of the video to analyze",
        min_length=10
    )
    court_type: Literal["singles", "doubles"] = Field(
        default="singles",
        description="Type of badminton court"
    )


class AnalyzeResponse(BaseModel):
    """Response schema for video analysis"""

    decision: Literal["IN", "OUT", "UNCERTAIN"] = Field(
        ...,
        description="Landing decision: IN, OUT, or UNCERTAIN"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    landing_x: float = Field(
        ...,
        description="X coordinate of shuttle landing point"
    )
    landing_y: float = Field(
        ...,
        description="Y coordinate of shuttle landing point"
    )
    landing_frame_url: str = Field(
        ...,
        description="S3 URL of the landing frame image"
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds"
    )
    frame_number: int = Field(
        default=0,
        description="Frame number where shuttle landed"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(default="healthy")
    service: str = Field(default="fairplay-cv-service")
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    """Error response schema"""

    error: str = Field(..., description="Error message")
    detail: str = Field(default="", description="Detailed error information")
