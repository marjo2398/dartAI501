import random
import math
from typing import Tuple
from vision.dartboard import get_score_from_canonical, ScoreResult, SECTOR_ORDER

# Standard target coordinates for sectors in canonical space
SECTOR_ANGLES = {
    20: 90, 1: 72, 18: 54, 4: 36, 13: 18, 6: 0, 10: -18, 15: -36, 2: -54, 17: -72,
    3: -90, 19: -108, 7: -126, 16: -144, 8: -162, 11: 180, 14: 162, 9: 144, 12: 126, 5: 108
}

SKILL_STD_DEV = {
    "beginner": 0.30,      # High spread
    "intermediate": 0.15,  # Moderate spread
    "advanced": 0.08,      # Low spread
    "pro": 0.03            # Very tight grouping
}

class AIBot:
    def __init__(self, name: str = "Autodarts Bot", skill_level: str = "intermediate"):
        self.name = name
        self.skill_level = skill_level if skill_level in SKILL_STD_DEV else "intermediate"
        self.std_dev = SKILL_STD_DEV[self.skill_level]

    def get_target_coordinates(self, current_score: int) -> Tuple[float, float]:
        """Determines target canonical (x, y) based on optimal X01 strategy."""
        if current_score > 60:
            # Aim for T20 (x=0, y=0.606)
            return (0.0, 0.606)
        elif current_score == 50:
            # Bullseye
            return (0.0, 0.0)
        elif current_score <= 40 and current_score % 2 == 0:
            # Aim for Double
            sector = current_score // 2
            angle_deg = SECTOR_ANGLES.get(sector, 90)
            rad = math.radians(angle_deg)
            # Double ring radius is ~0.97
            return (0.97 * math.cos(rad), 0.97 * math.sin(rad))
        else:
            # Aim for single to set up double
            sector = current_score - 32 if current_score > 32 else (current_score - 16 if current_score > 16 else 1)
            angle_deg = SECTOR_ANGLES.get(sector, 90)
            rad = math.radians(angle_deg)
            return (0.75 * math.cos(rad), 0.75 * math.sin(rad))

    def generate_throw(self, current_score: int) -> ScoreResult:
        """Simulates a bot throw adding Gaussian dispersion according to skill level."""
        target_x, target_y = self.get_target_coordinates(current_score)
        actual_x = random.gauss(target_x, self.std_dev)
        actual_y = random.gauss(target_y, self.std_dev)
        return get_score_from_canonical(actual_x, actual_y)
