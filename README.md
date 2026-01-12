# FairPlay Badminton - Computer Vision Microservice

Python-based microservice for analyzing badminton videos to determine shuttle landing points (IN/OUT decisions).

## Features

- Shuttle detection using YOLOv8 or classical CV methods
- Landing frame extraction
- IN/OUT decision with confidence scoring
- FastAPI REST API
- S3 integration for video and frame storage

## Tech Stack

- **FastAPI**: Web framework
- **OpenCV**: Video processing
- **YOLOv8** (optional): Object detection
- **boto3**: AWS S3 integration
- **Python 3.10+**

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run the service
python main.py
```

## API Endpoints

### POST /analyze

Analyzes a badminton video and returns IN/OUT decision.

**Request:**
```json
{
  "video_url": "https://s3.amazonaws.com/...",
  "court_type": "singles"
}
```

**Response:**
```json
{
  "decision": "IN",
  "confidence": 0.85,
  "landing_x": 245.5,
  "landing_y": 180.2,
  "landing_frame_url": "https://s3.amazonaws.com/.../landing_frame.jpg",
  "processing_time": 2.3
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "fairplay-cv-service",
  "version": "1.0.0"
}
```

## Architecture

```
FastAPI Server
  ↓
Video Download from S3
  ↓
OpenCV Processing Pipeline
  ↓
Shuttle Detection
  ↓
Landing Frame Detection
  ↓
IN/OUT Decision Algorithm
  ↓
Upload Landing Frame to S3
  ↓
Return JSON Response
```

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload --port 8000

# Run tests
pytest tests/

# Lint code
black .
flake8 .
```

## Deployment

- Can run on CPU (slower but cost-effective)
- GPU recommended for faster processing
- Deploy on Fly.io, Railway, or any Python-compatible platform

## Cost Optimization

- Process only disputed points (not full matches)
- Videos are short (10-20 seconds)
- CPU processing acceptable for MVP (2-5 second response time)
- GPU optional for scale
