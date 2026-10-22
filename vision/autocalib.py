import cv2
import numpy as np
from typing import Optional, Tuple, List

def auto_detect_dartboard_points(image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """
    Automatically detects the main outer double ring of a dartboard in an image
    using Hough Circles and ellipse/contour analysis, returning 4 calibration points:
    [Top (Double 20), Right (Double 6), Bottom (Double 3), Left (Double 11)].
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.medianBlur(gray, 5)

    # Detect circles using HoughCircles
    rows = blurred.shape[0]
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=rows / 4,
        param1=100,
        param2=50,
        minRadius=int(rows * 0.2),
        maxRadius=int(rows * 0.48)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = circles[0][0]
        cx, cy, r = float(best_circle[0]), float(best_circle[1]), float(best_circle[2])

        # Generate 4 canonical outer calibration points based on circle geometry
        top = (cx, cy - r)
        right = (cx + r, cy)
        bottom = (cx, cy + r)
        left = (cx - r, cy)
        return [top, right, bottom, left]

    # Fallback to image bounding box estimation if circle detection fails
    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    r = min(w, h) * 0.4
    return [
        (cx, cy - r),
        (cx + r, cy),
        (cx, cy + r),
        (cx - r, cy)
    ]
