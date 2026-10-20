import pytest
import time
from game.database import init_db, log_dart, log_vision_diagnostic
from analytics import AnalyticsEngine

def test_database_and_analytics():
    init_db()

    unique_player = f"Player_{int(time.time() * 1000)}"
    unique_match_id = int(time.time() * 1000)

    # Log 3 darts for unique player in turn 1
    log_dart(unique_match_id, unique_player, 1, 1, 1, "T20", 3, 20, 60, 0.01, 0.61)
    log_dart(unique_match_id, unique_player, 1, 1, 2, "T20", 3, 20, 60, 0.02, 0.60)
    log_dart(unique_match_id, unique_player, 1, 1, 3, "T20", 3, 20, 60, 0.00, 0.62)

    stats = AnalyticsEngine.get_player_summary_stats(unique_player)
    assert stats["total_darts"] == 3
    assert stats["overall_average_3dart"] == 180.0
    assert stats["first_9_average"] == 180.0
    assert stats["scoring_brackets"]["180"] == 1

    heatmap = AnalyticsEngine.get_heatmap_data(unique_player)
    assert len(heatmap) == 3

    # Log vision diagnostics
    log_vision_diagnostic(45.2, 12.5, True, 0.01, 0.61)
    diag = AnalyticsEngine.get_vision_diagnostics()
    assert diag["total_processed_frames"] >= 1
    assert diag["avg_detection_latency_ms"] > 0
