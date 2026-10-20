import math

# Standard sector order clockwise starting from top sector (20)
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Standard WDF dartboard dimensions in millimeters
DOUBLE_OUTER_RADIUS_MM = 170.0
DOUBLE_INNER_RADIUS_MM = 162.0
TRIPLE_OUTER_RADIUS_MM = 107.0
TRIPLE_INNER_RADIUS_MM = 99.0
OUTER_BULL_RADIUS_MM = 15.9
INNER_BULL_RADIUS_MM = 6.35

# Normalized radii (normalized such that DOUBLE_OUTER_RADIUS_MM = 1.0)
R_DOUBLE_OUTER = 1.0
R_DOUBLE_INNER = DOUBLE_INNER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM
R_TRIPLE_OUTER = TRIPLE_OUTER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM
R_TRIPLE_INNER = TRIPLE_INNER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM
R_OUTER_BULL = OUTER_BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM
R_INNER_BULL = INNER_BULL_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM


class ScoreResult:
    def __init__(self, name: str, multiplier: int, sector: int, value: int):
        self.name = name          # e.g., "T20", "D16", "S5", "BULL", "OUTER_BULL", "MISS"
        self.multiplier = multiplier  # 0, 1, 2, 3
        self.sector = sector      # 0..20, 25 for bull
        self.value = value        # Total points (e.g. 60, 32, 5, 50, 25, 0)

    def to_dict(self):
        return {
            "name": self.name,
            "multiplier": self.multiplier,
            "sector": self.sector,
            "value": self.value
        }

    def __repr__(self):
        return f"ScoreResult({self.name}, value={self.value})"


def get_score_from_canonical(x: float, y: float) -> ScoreResult:
    """
    Given (x, y) coordinates where (0,0) is Bullseye center and 1.0 is the outer edge of the Double ring:
    - positive x is right
    - positive y is top (upwards)
    Returns ScoreResult.
    """
    r = math.hypot(x, y)

    if r <= R_INNER_BULL:
        return ScoreResult("BULL", 2, 25, 50)
    elif r <= R_OUTER_BULL:
        return ScoreResult("OUTER_BULL", 1, 25, 25)
    elif r > R_DOUBLE_OUTER:
        return ScoreResult("MISS", 0, 0, 0)

    # Calculate angle in degrees from positive X axis (0 to 360)
    # Note: Sector 20 is centered around +90 degrees (top / positive Y)
    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad) % 360.0

    # Sector 20 spans from 81° to 99° in standard 2D cartesian (Y up, X right)
    # Rotate coordinate system by -81° so Sector 20 starts at 0°
    rotated_deg = (angle_deg - 81.0) % 360.0
    sector_index = int(rotated_deg // 18)
    sector_num = SECTOR_ORDER[sector_index]

    if R_TRIPLE_INNER <= r <= R_TRIPLE_OUTER:
        return ScoreResult(f"T{sector_num}", 3, sector_num, sector_num * 3)
    elif R_DOUBLE_INNER <= r <= R_DOUBLE_OUTER:
        return ScoreResult(f"D{sector_num}", 2, sector_num, sector_num * 2)
    else:
        return ScoreResult(f"S{sector_num}", 1, sector_num, sector_num)
