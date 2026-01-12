"""
FairPlay Badminton - Computer Vision Microservice
FastAPI server for analyzing badminton videos
"""
import logging
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, ErrorResponse
from video_processor_enhanced import EnhancedVideoProcessor
from s3_utils import S3Handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FairPlay CV Service",
    description="Computer vision microservice for badminton IN/OUT decisions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
video_processor = EnhancedVideoProcessor()
s3_handler = S3Handler()

# Temporary directories
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="fairplay-cv-service",
        version="1.0.0"
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    """
    Analyze badminton video and return IN/OUT decision

    Process:
    1. Download video from S3
    2. Detect shuttle and landing frame
    3. Determine IN/OUT decision
    4. Upload landing frame to S3
    5. Return results
    """
    start_time = time.time()
    video_id = str(uuid.uuid4())[:8]

    logger.info(f"[{video_id}] Analyzing video: {request.video_url[:50]}...")

    # Temporary file paths
    video_path = TEMP_DIR / f"{video_id}_video.mp4"
    frame_path = TEMP_DIR / f"{video_id}_landing.jpg"

    try:
        # Step 1: Download video from S3
        logger.info(f"[{video_id}] Downloading video from S3...")
        success = s3_handler.download_video(request.video_url, str(video_path))

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to download video from S3"
            )

        # Step 2: Process video
        logger.info(f"[{video_id}] Processing video...")
        result = video_processor.process_video(
            str(video_path),
            request.court_type
        )

        # Step 3: Save landing frame
        logger.info(f"[{video_id}] Saving landing frame...")
        video_processor.save_landing_frame(result["landing_frame"], str(frame_path))

        # Step 4: Upload landing frame to S3
        logger.info(f"[{video_id}] Uploading landing frame to S3...")
        s3_key = f"landing_frames/{video_id}_landing.jpg"
        frame_url = s3_handler.upload_frame(str(frame_path), s3_key)

        if not frame_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload landing frame to S3"
            )

        # Step 5: Calculate processing time
        processing_time = time.time() - start_time

        # Step 6: Return results
        response = AnalyzeResponse(
            decision=result["decision"],
            confidence=result["confidence"],
            landing_x=result["landing_x"],
            landing_y=result["landing_y"],
            landing_frame_url=frame_url,
            processing_time=round(processing_time, 2),
            frame_number=result["frame_number"]
        )

        logger.info(
            f"[{video_id}] Analysis complete: {response.decision} "
            f"(confidence: {response.confidence:.2f}, time: {processing_time:.2f}s)"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{video_id}] Error processing video: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

    finally:
        # Cleanup temporary files
        try:
            if video_path.exists():
                video_path.unlink()
            if frame_path.exists():
                frame_path.unlink()
        except Exception as e:
            logger.warning(f"[{video_id}] Error cleaning up temp files: {e}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FairPlay CV Service...")
    logger.info(f"Service will run on {settings.service_host}:{settings.service_port}")

    uvicorn.run(
        "main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=settings.debug,
        log_level="info"
    )
