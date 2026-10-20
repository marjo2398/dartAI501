import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from game.database import get_db_connection

class AnalyticsEngine:
    @staticmethod
    def get_player_summary_stats(player_name: Optional[str] = None) -> Dict[str, Any]:
        """Calculates professional dart statistics for a given player or all players."""
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM darts"
        params = []
        if player_name:
            query += " WHERE player_name = ?"
            params.append(player_name)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                "total_darts": 0,
                "overall_average_3dart": 0.0,
                "first_9_average": 0.0,
                "checkout_percentage": 0.0,
                "double_attempts": 0,
                "double_hits": 0,
                "scoring_brackets": {"180": 0, "140+": 0, "100+": 0, "60+": 0, "<60": 0},
                "directional_bias_mm": {"x_drift": 0.0, "y_drift": 0.0, "grouping_radius_mm": 0.0}
            }

        total_darts = len(rows)
        total_points = sum(r["score_value"] for r in rows)
        avg_3dart = round((total_points / total_darts) * 3, 2) if total_darts > 0 else 0.0

        # First 9 darts average calculation (darts 1..9 in each leg)
        first_9_darts = [r["score_value"] for r in rows if r["dart_number_in_turn"] is not None and r["turn_number"] is not None and r["turn_number"] <= 3]
        first_9_avg = round((sum(first_9_darts) / len(first_9_darts)) * 3, 2) if first_9_darts else 0.0

        # Double checkout stats
        double_attempts = sum(1 for r in rows if r["is_double_attempt"])
        double_hits = sum(1 for r in rows if r["is_checkout_hit"])
        checkout_pct = round((double_hits / double_attempts) * 100, 1) if double_attempts > 0 else 0.0

        # Scoring Brackets (grouping by 3-dart turn totals)
        turns: Dict[Tuple[Any, Any, Any], List[int]] = {}
        for r in rows:
            # Fallback if match_id/leg_number/turn_number are missing
            m_id = r["match_id"] if r["match_id"] is not None else 1
            leg_num = r["leg_number"] if r["leg_number"] is not None else 1
            turn_num = r["turn_number"] if r["turn_number"] is not None else 1
            key = (m_id, leg_num, turn_num)
            if key not in turns:
                turns[key] = []
            turns[key].append(r["score_value"])

        brackets = {"180": 0, "140+": 0, "100+": 0, "60+": 0, "<60": 0}
        for turn_scores in turns.values():
            t_sum = sum(turn_scores)
            if t_sum == 180:
                brackets["180"] += 1
            elif t_sum >= 140:
                brackets["140+"] += 1
            elif t_sum >= 100:
                brackets["100+"] += 1
            elif t_sum >= 60:
                brackets["60+"] += 1
            else:
                brackets["<60"] += 1

        # Directional Drift Bias (aiming for T20 vs actual impact)
        # T20 canonical location is approx (x=0, y=0.606)
        x_offsets = []
        y_offsets = []
        for r in rows:
            if r["canonical_x"] is not None and r["canonical_y"] is not None:
                # Assuming standard aim is T20 (x=0, y=0.606)
                dx = r["canonical_x"] - 0.0
                dy = r["canonical_y"] - 0.606
                x_offsets.append(dx)
                y_offsets.append(dy)

        x_drift_mm = round(float(np.mean(x_offsets)) * 170.0, 1) if x_offsets else 0.0
        y_drift_mm = round(float(np.mean(y_offsets)) * 170.0, 1) if y_offsets else 0.0
        grouping_radius_mm = round(float(np.mean(np.hypot(x_offsets, y_offsets))) * 170.0, 1) if x_offsets else 0.0

        return {
            "total_darts": total_darts,
            "overall_average_3dart": avg_3dart,
            "first_9_average": first_9_avg,
            "checkout_percentage": checkout_pct,
            "double_attempts": double_attempts,
            "double_hits": double_hits,
            "scoring_brackets": brackets,
            "directional_bias_mm": {
                "x_drift_mm": x_drift_mm,
                "y_drift_mm": y_drift_mm,
                "grouping_radius_mm": grouping_radius_mm
            }
        }

    @staticmethod
    def get_heatmap_data(player_name: Optional[str] = None) -> List[Dict[str, float]]:
        """Returns canonical (x, y) coordinates for heatmap / scatter visualization."""
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT canonical_x, canonical_y, score_value FROM darts WHERE canonical_x IS NOT NULL"
        params = []
        if player_name:
            query += " AND player_name = ?"
            params.append(player_name)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [{"x": float(r["canonical_x"]), "y": float(r["canonical_y"]), "value": int(r["score_value"])} for r in rows]

    @staticmethod
    def get_vision_diagnostics() -> Dict[str, Any]:
        """Calculates Computer Vision Engine diagnostics (latency, frame diffs, success rate)."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vision_logs ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                "avg_detection_latency_ms": 0.0,
                "avg_diff_intensity": 0.0,
                "success_rate_pct": 0.0,
                "total_processed_frames": 0
            }

        latencies = [r["detection_duration_ms"] for r in rows if r["detection_duration_ms"] is not None]
        diffs = [r["diff_mean_intensity"] for r in rows if r["diff_mean_intensity"] is not None]
        detections = [1 for r in rows if r["detected_dart"]]

        return {
            "avg_detection_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "avg_diff_intensity": round(float(np.mean(diffs)), 2) if diffs else 0.0,
            "success_rate_pct": round((len(detections) / len(rows)) * 100, 1) if rows else 0.0,
            "total_processed_frames": len(rows)
        }
