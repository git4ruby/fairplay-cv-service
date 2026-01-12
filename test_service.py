"""
Simple test script for the CV service
"""
import requests
import json

# Service URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_analyze():
    """Test analyze endpoint with a sample video URL"""
    print("Testing /analyze endpoint...")

    # You'll need to replace this with an actual S3 video URL
    test_data = {
        "video_url": "https://fairplay-badminton-videos.s3.us-east-1.amazonaws.com/test_rally.mp4",
        "court_type": "singles"
    }

    response = requests.post(
        f"{BASE_URL}/analyze",
        json=test_data
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        print(f"\nDecision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Landing position: ({result['landing_x']:.1f}, {result['landing_y']:.1f})")
        print(f"Processing time: {result['processing_time']:.2f}s")
    else:
        print(f"Error: {response.text}")

    print()


if __name__ == "__main__":
    print("FairPlay CV Service - Test Script")
    print("=" * 50)
    print()

    try:
        # Test health endpoint
        test_health()

        # Test analyze endpoint
        # test_analyze()  # Uncomment when you have a video URL to test

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to service. Make sure it's running on port 8000")
    except Exception as e:
        print(f"Error: {e}")
