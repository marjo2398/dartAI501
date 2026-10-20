import cv2
import numpy as np
from typing import List, Tuple, Optional
from .dartboard import get_score_from_canonical, ScoreResult

# Canonical calibration target reference points in normalized coordinates [-1, 1]
# Standard order for 4-point calibration:
# 1. Top - Outer Double 20 (x=0, y=1.0)
# 2. Right - Outer Double 6 (x=1.0, y=0)
# 3. Bottom - Outer Double 3 (x=0, y=-1.0)
# 4. Left - Outer Double 11 (x=-1.0, y=0)
CANONICAL_CALIB_POINTS = np.array([
    [0.0, 1.0],   # Top (20)
    [1.0, 0.0],   # Right (6)
    [0.0, -1.0],  # Bottom (3)
    [-1.0, 0.0]   # Left (11)
], dtype=np.float32)

class CalibrationEngine:
    def __init__(self):
        self.H: Optional[np.ndarray] = None  # Homography matrix from camera pixels to canonical
        self.H_inv: Optional[np.ndarray] = None
        self.points_image: List[Tuple[float, float]] = []

    def set_calibration_points(self, image_points: List[Tuple[float, float]]) -> bool:
        """
        Calculates homography matrix given 4 calibration points in image coordinates:
        Points order: [Top(Double 20), Right(Double 6), Bottom(Double 3), Left(Double 11)]
        OR center + 4 calibration points.
        If 4 points are passed, uses CANONICAL_CALIB_POINTS.
        """
        if len(image_points) != 4:
            return False

        pts_src = np.array(image_points, dtype=np.float32)
        pts_dst = CANONICAL_CALIB_POINTS

        self.H, _ = cv2.findHomography(pts_src, pts_dst)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            self.points_image = image_points
            return True
        return False

    def transform_pixel_to_canonical(self, px: float, py: float) -> Tuple[float, float]:
        """Transforms image pixel coordinates (px, py) to canonical coordinates (cx, cy)."""
        if self.H is None:
            raise ValueError("Calibration not set.")
        pt = np.array([px, py, 1.0], dtype=np.float32)
        transformed = np.dot(self.H, pt)
        if transformed[2] != 0:
            cx = transformed[0] / transformed[2]
            cy = transformed[1] / transformed[2]
            return float(cx), float(cy)
        return 0.0, 0.0

    def get_score_for_pixel(self, px: float, py: float) -> ScoreResult:
        cx, cy = self.transform_pixel_to_canonical(px, py)
        return get_score_from_canonical(cx, cy)


class DartDetector:
    def __init__(self, calibration: CalibrationEngine):
        self.calibration = calibration

    def detect_dart_impact(self, img_before: np.ndarray, img_after: np.ndarray, min_area: int = 15) -> Optional[Tuple[float, float, ScoreResult]]:
        """
        Compares frame before throw and frame after throw using image subtraction.
        Detects newly introduced dart point location in image pixel coordinates,
        and converts it to canonical score.
        """
        if img_before is None or img_after is None:
            return None

        # Convert to grayscale
        gray1 = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY) if len(img_before.shape) == 3 else img_before
        gray2 = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY) if len(img_after.shape) == 3 else img_after

        # Blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # Absolute difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find largest contour corresponding to the inserted dart
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            return None

        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Find the dart tip (closest point of contour to the center of dartboard in pixel space)
        # Transform board center (0,0) back to pixel space if calibration is set
        if self.calibration.H_inv is not None:
            center_pt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            pt_px = np.dot(self.calibration.H_inv, center_pt)
            center_px = (pt_px[0] / pt_px[2], pt_px[1] / pt_px[2])
        else:
            # Fallback to image center
            center_px = (img_after.shape[1] / 2.0, img_after.shape[0] / 2.0)

        # Dart tip is typically the contour point closest to the board surface / center
        pts = largest_contour.reshape(-1, 2)
        dists = np.linalg.norm(pts - center_px, axis=1)
        tip_index = np.argmin(dists)
        tip_px, tip_py = float(pts[tip_index][0]), float(pts[tip_index][1])

        score = self.calibration.get_score_for_pixel(tip_px, tip_py)
        return tip_px, tip_py, score
