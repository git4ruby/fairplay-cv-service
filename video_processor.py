"""
Core video processing logic using OpenCV
Handles shuttle detection, landing frame extraction, and IN/OUT decisions
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VideoProcessor:
    """Processes badminton videos to detect shuttle and determine IN/OUT"""

    def __init__(self):
        self.confidence_threshold = settings.confidence_threshold
        self.shuttle_min_confidence = settings.shuttle_min_confidence
        self.frame_skip = settings.frame_skip

    def process_video(self, video_path: str, court_type: str = "singles") -> Dict:
        """
        Main processing function for badminton video

        Args:
            video_path: Path to video file
            court_type: "singles" or "doubles"

        Returns:
            Dict with decision, confidence, landing coordinates, frame number
        """
        logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video: {total_frames} frames, {fps} fps, {width}x{height}")

            # Detect shuttle trajectory and landing frame
            landing_frame_num, landing_x, landing_y, confidence = self._detect_shuttle_landing(
                cap, total_frames, fps
            )

            # Extract landing frame
            landing_frame = self._extract_frame(video_path, landing_frame_num)

            # Determine IN/OUT decision based on court boundaries
            decision = self._determine_in_out(
                landing_x, landing_y, width, height, court_type
            )

            result = {
                "decision": decision,
                "confidence": confidence,
                "landing_x": landing_x,
                "landing_y": landing_y,
                "frame_number": landing_frame_num,
                "landing_frame": landing_frame,
            }

            logger.info(f"Processing complete: {decision} at ({landing_x}, {landing_y})")
            return result

        finally:
            cap.release()

    def _detect_shuttle_landing(
        self, cap: cv2.VideoCapture, total_frames: int, fps: float
    ) -> Tuple[int, float, float, float]:
        """
        Detect shuttle trajectory and landing frame

        Strategy:
        1. Use background subtraction to detect moving objects
        2. Track small white objects (shuttle)
        3. Detect when shuttle stops moving (landing)
        4. Return landing frame and coordinates

        Returns:
            (frame_number, x, y, confidence)
        """
        # Background subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        shuttle_positions = []
        frame_num = 0

        # Process frames to detect shuttle
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for faster processing
            if frame_num % self.frame_skip != 0:
                frame_num += 1
                continue

            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)

            # Find contours (potential shuttle)
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Look for small white circular objects (shuttle characteristics)
            for contour in contours:
                area = cv2.contourArea(contour)

                # Shuttle is typically small (10-100 pixels area depending on distance)
                if 10 < area < 200:
                    # Get center point
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        shuttle_positions.append({
                            "frame": frame_num,
                            "x": cx,
                            "y": cy,
                            "area": area
                        })

            frame_num += 1

        # Analyze shuttle trajectory to find landing
        if len(shuttle_positions) < 5:
            # Not enough data, use middle of last frames
            logger.warning("Insufficient shuttle detections, using fallback")
            return self._fallback_detection(cap, total_frames)

        # Find landing point: where shuttle stops moving significantly
        landing_idx = self._find_landing_point(shuttle_positions)

        if landing_idx >= 0:
            landing = shuttle_positions[landing_idx]
            confidence = self._calculate_confidence(shuttle_positions, landing_idx)

            return (
                landing["frame"],
                float(landing["x"]),
                float(landing["y"]),
                confidence
            )
        else:
            # Fallback to last detected position
            landing = shuttle_positions[-1]
            return (
                landing["frame"],
                float(landing["x"]),
                float(landing["y"]),
                0.5  # Medium confidence for fallback
            )

    def _find_landing_point(self, positions: list) -> int:
        """
        Analyze shuttle positions to find where it lands

        Landing detected when:
        - Vertical movement stops (y position stabilizes)
        - Horizontal movement slows down
        - Position stays relatively constant

        Returns:
            Index of landing position, or -1 if not found
        """
        if len(positions) < 5:
            return -1

        # Calculate movement between consecutive positions
        movements = []
        for i in range(1, len(positions)):
            dx = abs(positions[i]["x"] - positions[i-1]["x"])
            dy = abs(positions[i]["y"] - positions[i-1]["y"])
            movement = np.sqrt(dx**2 + dy**2)
            movements.append(movement)

        # Find where movement drops significantly (shuttle lands)
        avg_movement = np.mean(movements)
        threshold = avg_movement * 0.3  # 30% of average movement

        for i in range(len(movements) - 3):
            # Check if next 3 movements are all below threshold
            if all(movements[i+j] < threshold for j in range(3)):
                return i + 1  # Return position where landing started

        # If no clear landing, return last position
        return len(positions) - 1

    def _calculate_confidence(self, positions: list, landing_idx: int) -> float:
        """
        Calculate confidence score for landing detection

        Higher confidence when:
        - More shuttle positions detected
        - Clear trajectory pattern
        - Stable landing point
        """
        # Base confidence on number of detections
        detection_confidence = min(len(positions) / 20.0, 1.0)  # Max at 20 detections

        # Check trajectory smoothness before landing
        trajectory_confidence = 0.7  # Default

        if landing_idx > 3:
            # Check if positions before landing follow smooth trajectory
            pre_landing = positions[max(0, landing_idx-5):landing_idx]
            if len(pre_landing) >= 3:
                # Check y-direction consistency (shuttle should be moving down)
                y_diffs = [pre_landing[i+1]["y"] - pre_landing[i]["y"]
                          for i in range(len(pre_landing)-1)]
                if all(d > -5 for d in y_diffs):  # Mostly downward or stable
                    trajectory_confidence = 0.9

        # Overall confidence
        confidence = (detection_confidence * 0.6 + trajectory_confidence * 0.4)
        return min(max(confidence, 0.0), 1.0)

    def _fallback_detection(
        self, cap: cv2.VideoCapture, total_frames: int
    ) -> Tuple[int, float, float, float]:
        """
        Fallback detection when shuttle tracking fails

        Use simple heuristic: assume landing is in last 20% of video,
        look for bright spots in lower half of frame
        """
        logger.info("Using fallback detection method")

        # Go to 80% through video
        landing_frame = int(total_frames * 0.8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, landing_frame)

        ret, frame = cap.read()
        if not ret:
            landing_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, landing_frame)
            ret, frame = cap.read()

        height, width = frame.shape[:2]

        # Look for bright spots in lower half
        lower_half = frame[height//2:, :]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find brightest spot
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest bright contour
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + height // 2  # Adjust for lower half

                return landing_frame, float(cx), float(cy), 0.4

        # Ultimate fallback: center-bottom of frame
        return landing_frame, float(width // 2), float(height * 0.7), 0.3

    def _extract_frame(self, video_path: str, frame_number: int) -> np.ndarray:
        """Extract specific frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Cannot extract frame {frame_number}")

        return frame

    def _determine_in_out(
        self, x: float, y: float, width: int, height: int, court_type: str
    ) -> str:
        """
        Determine if shuttle landed IN or OUT based on court boundaries

        Simplified approach:
        - Assumes court occupies central 80% of frame width, 70% of height
        - Singles court: narrower (60% width)
        - Doubles court: wider (80% width)

        TODO: In production, use court line detection with OpenCV Hough Transform
        """
        # Court boundary estimation (percentage of frame)
        if court_type == "singles":
            court_width_pct = 0.6
        else:
            court_width_pct = 0.8

        court_height_pct = 0.7

        # Calculate court boundaries
        court_left = width * (1 - court_width_pct) / 2
        court_right = width - court_left
        court_top = height * (1 - court_height_pct) / 2
        court_bottom = height - court_top

        # Check if landing point is inside court
        in_horizontal = court_left <= x <= court_right
        in_vertical = court_top <= y <= court_bottom

        if in_horizontal and in_vertical:
            return "IN"
        else:
            return "OUT"

    def save_landing_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """
        Save landing frame as JPEG with annotations

        Args:
            frame: Landing frame image
            output_path: Path to save image

        Returns:
            True if successful
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save frame
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            logger.info(f"Landing frame saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
