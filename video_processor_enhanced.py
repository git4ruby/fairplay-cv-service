"""
Enhanced video processing with YOLOv8 and court line detection
Provides more accurate shuttle detection and IN/OUT decisions
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Try to import YOLOv8 (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLOv8 is available for shuttle detection")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not available, will use classical CV methods")


class EnhancedVideoProcessor:
    """Enhanced processor with YOLOv8 and court line detection"""

    def __init__(self):
        self.confidence_threshold = settings.confidence_threshold
        self.shuttle_min_confidence = settings.shuttle_min_confidence
        self.frame_skip = settings.frame_skip

        # Initialize YOLOv8 model if available
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Start with YOLOv8n (nano) model - fast and lightweight
                # For badminton-specific detection, you can fine-tune this model
                self.yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv8 model: {e}")
                self.yolo_model = None

    def process_video(self, video_path: str, court_type: str = "singles") -> Dict:
        """
        Main processing function with enhanced detection

        Args:
            video_path: Path to video file
            court_type: "singles" or "doubles"

        Returns:
            Dict with decision, confidence, landing coordinates, frame number
        """
        logger.info(f"Processing video with enhanced detection: {video_path}")

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

            # Step 1: Detect court lines for accurate boundaries
            court_lines = self._detect_court_lines(cap, width, height)
            logger.info(f"Court lines detected: {len(court_lines)} lines")

            # Step 2: Detect shuttle trajectory and landing frame
            if self.yolo_model:
                landing_frame_num, landing_x, landing_y, confidence = self._detect_shuttle_yolo(
                    cap, total_frames, fps
                )
            else:
                landing_frame_num, landing_x, landing_y, confidence = self._detect_shuttle_classical(
                    cap, total_frames, fps
                )

            # Step 3: Extract landing frame
            landing_frame = self._extract_frame(video_path, landing_frame_num)

            # Step 4: Determine IN/OUT based on detected court lines
            decision, adjusted_confidence = self._determine_in_out_advanced(
                landing_x, landing_y, court_lines, court_type, confidence
            )

            result = {
                "decision": decision,
                "confidence": adjusted_confidence,
                "landing_x": landing_x,
                "landing_y": landing_y,
                "frame_number": landing_frame_num,
                "landing_frame": landing_frame,
            }

            logger.info(
                f"Enhanced processing complete: {decision} at ({landing_x:.1f}, {landing_y:.1f}) "
                f"confidence: {adjusted_confidence:.2f}"
            )
            return result

        finally:
            cap.release()

    def _detect_court_lines(
        self, cap: cv2.VideoCapture, width: int, height: int, num_samples: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect court lines using Hough Transform

        Strategy:
        1. Sample multiple frames from video
        2. Apply edge detection
        3. Use Hough Line Transform to find straight lines
        4. Filter and merge similar lines
        5. Return court boundary lines

        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        logger.info("Detecting court lines with Hough Transform...")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_lines = []

        # Sample frames evenly throughout the video
        sample_frames = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Apply Hough Line Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=int(width * 0.3),  # Lines must be at least 30% of width
                maxLineGap=50
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    all_lines.append((x1, y1, x2, y2))

        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not all_lines:
            logger.warning("No court lines detected, using default boundaries")
            return []

        # Merge similar lines
        merged_lines = self._merge_similar_lines(all_lines, width, height)

        # Filter to keep only court boundary lines (long horizontal/vertical lines)
        court_lines = self._filter_court_lines(merged_lines, width, height)

        logger.info(f"Detected {len(court_lines)} court boundary lines")
        return court_lines

    def _merge_similar_lines(
        self, lines: List[Tuple[int, int, int, int]], width: int, height: int, threshold: float = 20
    ) -> List[Tuple[int, int, int, int]]:
        """Merge lines that are very similar in position and angle"""
        if not lines:
            return []

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            x1, y1, x2, y2 = line1
            similar_lines = [line1]

            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue

                x3, y3, x4, y4 = line2

                # Check if lines are similar (close endpoints)
                dist = min(
                    np.sqrt((x1-x3)**2 + (y1-y3)**2),
                    np.sqrt((x1-x4)**2 + (y1-y4)**2),
                    np.sqrt((x2-x3)**2 + (y2-y3)**2),
                    np.sqrt((x2-x4)**2 + (y2-y4)**2)
                )

                if dist < threshold:
                    similar_lines.append(line2)
                    used.add(j)

            # Average similar lines
            avg_x1 = int(np.mean([l[0] for l in similar_lines]))
            avg_y1 = int(np.mean([l[1] for l in similar_lines]))
            avg_x2 = int(np.mean([l[2] for l in similar_lines]))
            avg_y2 = int(np.mean([l[3] for l in similar_lines]))

            merged.append((avg_x1, avg_y1, avg_x2, avg_y2))
            used.add(i)

        return merged

    def _filter_court_lines(
        self, lines: List[Tuple[int, int, int, int]], width: int, height: int
    ) -> List[Tuple[int, int, int, int]]:
        """Filter to keep only major court boundary lines"""
        court_lines = []

        for x1, y1, x2, y2 in lines:
            # Calculate line length and angle
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Keep only long lines (at least 30% of frame dimension)
            min_length = min(width, height) * 0.3
            if length < min_length:
                continue

            # Categorize as horizontal or vertical
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Horizontal lines (0-30 degrees or 150-180 degrees)
            is_horizontal = angle < 30 or angle > 150
            # Vertical lines (60-120 degrees)
            is_vertical = 60 < angle < 120

            if is_horizontal or is_vertical:
                court_lines.append((x1, y1, x2, y2))

        return court_lines

    def _detect_shuttle_yolo(
        self, cap: cv2.VideoCapture, total_frames: int, fps: float
    ) -> Tuple[int, float, float, float]:
        """
        Detect shuttle using YOLOv8 object detection

        YOLOv8 provides better detection than background subtraction,
        especially for fast-moving objects like shuttles
        """
        logger.info("Using YOLOv8 for shuttle detection")

        shuttle_positions = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for faster processing
            if frame_num % self.frame_skip != 0:
                frame_num += 1
                continue

            # Run YOLOv8 inference
            results = self.yolo_model(frame, verbose=False)

            # Look for small white objects (shuttle characteristics)
            # In a production system, you'd fine-tune YOLO on badminton shuttles
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    # Filter for small objects with high confidence
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    # Shuttle is small (adjust these thresholds based on your videos)
                    if 10 < area < 500 and conf > self.shuttle_min_confidence:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        shuttle_positions.append({
                            "frame": frame_num,
                            "x": cx,
                            "y": cy,
                            "confidence": conf,
                            "area": area
                        })

            frame_num += 1

        # Analyze trajectory to find landing
        if len(shuttle_positions) < 5:
            logger.warning("Insufficient YOLO detections, falling back to classical method")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self._detect_shuttle_classical(cap, total_frames, fps)

        landing_idx = self._find_landing_point(shuttle_positions)

        if landing_idx >= 0:
            landing = shuttle_positions[landing_idx]
            confidence = self._calculate_confidence_enhanced(shuttle_positions, landing_idx)

            return (
                landing["frame"],
                float(landing["x"]),
                float(landing["y"]),
                confidence
            )
        else:
            landing = shuttle_positions[-1]
            return (
                landing["frame"],
                float(landing["x"]),
                float(landing["y"]),
                landing.get("confidence", 0.6)
            )

    def _detect_shuttle_classical(
        self, cap: cv2.VideoCapture, total_frames: int, fps: float
    ) -> Tuple[int, float, float, float]:
        """
        Classical CV shuttle detection using background subtraction
        Fallback when YOLOv8 is not available
        """
        logger.info("Using classical CV for shuttle detection")

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        shuttle_positions = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % self.frame_skip != 0:
                frame_num += 1
                continue

            fg_mask = bg_subtractor.apply(frame)
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if 10 < area < 200:
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

        if len(shuttle_positions) < 5:
            logger.warning("Insufficient detections, using fallback")
            return self._fallback_detection(cap, total_frames)

        landing_idx = self._find_landing_point(shuttle_positions)

        if landing_idx >= 0:
            landing = shuttle_positions[landing_idx]
            confidence = self._calculate_confidence_enhanced(shuttle_positions, landing_idx)

            return (
                landing["frame"],
                float(landing["x"]),
                float(landing["y"]),
                confidence
            )
        else:
            landing = shuttle_positions[-1]
            return (
                landing["frame"],
                float(landing["x"]),
                float(landing["y"]),
                0.5
            )

    def _find_landing_point(self, positions: list) -> int:
        """Find where shuttle lands by analyzing movement"""
        if len(positions) < 5:
            return -1

        movements = []
        for i in range(1, len(positions)):
            dx = abs(positions[i]["x"] - positions[i-1]["x"])
            dy = abs(positions[i]["y"] - positions[i-1]["y"])
            movement = np.sqrt(dx**2 + dy**2)
            movements.append(movement)

        avg_movement = np.mean(movements)
        threshold = avg_movement * 0.3

        for i in range(len(movements) - 3):
            if all(movements[i+j] < threshold for j in range(3)):
                return i + 1

        return len(positions) - 1

    def _calculate_confidence_enhanced(self, positions: list, landing_idx: int) -> float:
        """Enhanced confidence calculation"""
        detection_confidence = min(len(positions) / 20.0, 1.0)
        trajectory_confidence = 0.7

        if landing_idx > 3:
            pre_landing = positions[max(0, landing_idx-5):landing_idx]
            if len(pre_landing) >= 3:
                y_diffs = [pre_landing[i+1]["y"] - pre_landing[i]["y"]
                          for i in range(len(pre_landing)-1)]
                if all(d > -5 for d in y_diffs):
                    trajectory_confidence = 0.9

        # Boost confidence if using YOLO detections
        if positions and "confidence" in positions[0]:
            avg_yolo_conf = np.mean([p.get("confidence", 0.5) for p in positions])
            trajectory_confidence = (trajectory_confidence + avg_yolo_conf) / 2

        confidence = (detection_confidence * 0.6 + trajectory_confidence * 0.4)
        return min(max(confidence, 0.0), 1.0)

    def _determine_in_out_advanced(
        self,
        x: float,
        y: float,
        court_lines: List[Tuple[int, int, int, int]],
        court_type: str,
        base_confidence: float
    ) -> Tuple[str, float]:
        """
        Advanced IN/OUT determination using detected court lines

        If court lines detected: Use actual line positions
        Otherwise: Fall back to estimated boundaries
        """
        if court_lines:
            return self._check_against_court_lines(x, y, court_lines, court_type, base_confidence)
        else:
            # Fallback to estimated boundaries
            logger.warning("Using estimated court boundaries (no lines detected)")
            return self._determine_in_out_estimated(x, y, court_type, base_confidence)

    def _check_against_court_lines(
        self,
        x: float,
        y: float,
        court_lines: List[Tuple[int, int, int, int]],
        court_type: str,
        base_confidence: float
    ) -> Tuple[str, float]:
        """Check if landing point is inside court based on detected lines"""

        # Find bounding box from court lines
        horizontal_lines = []
        vertical_lines = []

        for x1, y1, x2, y2 in court_lines:
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 30 or angle > 150:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif 60 < angle < 120:  # Vertical
                vertical_lines.append((x1, y1, x2, y2))

        if not horizontal_lines or not vertical_lines:
            return self._determine_in_out_estimated(x, y, court_type, base_confidence)

        # Find court boundaries
        all_x = []
        all_y = []
        for x1, y1, x2, y2 in court_lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        left = min(all_x)
        right = max(all_x)
        top = min(all_y)
        bottom = max(all_y)

        # Check if point is inside
        in_horizontal = left <= x <= right
        in_vertical = top <= y <= bottom

        # Calculate distance to nearest line for confidence adjustment
        min_dist = float('inf')
        for x1, y1, x2, y2 in court_lines:
            dist = self._point_to_line_distance(x, y, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)

        # If very close to line (< 10 pixels), reduce confidence
        if min_dist < 10:
            adjusted_confidence = base_confidence * 0.8
            decision = "UNCERTAIN"
        elif in_horizontal and in_vertical:
            adjusted_confidence = base_confidence * 1.1  # Boost confidence
            decision = "IN"
        else:
            adjusted_confidence = base_confidence * 1.1
            decision = "OUT"

        return decision, min(adjusted_confidence, 1.0)

    def _point_to_line_distance(
        self, px: float, py: float, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """Calculate perpendicular distance from point to line segment"""
        # Line segment length
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2

        if line_len_sq == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # Parameter t of projection point on line
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))

        # Projection point
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        # Distance from point to projection
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def _determine_in_out_estimated(
        self, x: float, y: float, court_type: str, base_confidence: float
    ) -> Tuple[str, float]:
        """Fallback method using estimated boundaries"""
        # This is a placeholder - implement based on frame dimensions
        # For now, assume center 80% x 70% is court
        # You would get actual dimensions from video
        return ("IN" if 0.1 < x / 1000 < 0.9 and 0.15 < y / 1000 < 0.85 else "OUT", base_confidence * 0.7)

    def _fallback_detection(
        self, cap: cv2.VideoCapture, total_frames: int
    ) -> Tuple[int, float, float, float]:
        """Fallback when all detection methods fail"""
        logger.info("Using fallback detection")

        landing_frame = int(total_frames * 0.8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, landing_frame)

        ret, frame = cap.read()
        if not ret:
            landing_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, landing_frame)
            ret, frame = cap.read()

        height, width = frame.shape[:2]
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

    def save_landing_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """Save landing frame as JPEG"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Landing frame saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
