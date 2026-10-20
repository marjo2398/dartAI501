import pytest
import numpy as np
import cv2
from vision.dartboard import get_score_from_canonical, R_INNER_BULL, R_OUTER_BULL
from vision.detector import CalibrationEngine, DartDetector
from game.x01 import X01Game, ScoreResult

def test_dartboard_geometry():
    # Bullseye center
    bull = get_score_from_canonical(0.0, 0.0)
    assert bull.name == "BULL"
    assert bull.value == 50

    # Outer Bull
    outer_bull = get_score_from_canonical(R_INNER_BULL + 0.01, 0.0)
    assert outer_bull.name == "OUTER_BULL"
    assert outer_bull.value == 25

    # Triple 20
    t20 = get_score_from_canonical(0.0, 0.606)
    assert t20.name == "T20"
    assert t20.value == 60

    # Double 20
    d20 = get_score_from_canonical(0.0, 0.98)
    assert d20.name == "D20"
    assert d20.value == 40

    # Single 20
    s20 = get_score_from_canonical(0.0, 0.3)
    assert s20.name == "S20"
    assert s20.value == 20

    # Miss
    miss = get_score_from_canonical(1.5, 1.5)
    assert miss.name == "MISS"
    assert miss.value == 0

def test_calibration_engine():
    calib = CalibrationEngine()
    pts = [(100.0, 20.0), (180.0, 100.0), (100.0, 180.0), (20.0, 100.0)]
    assert calib.set_calibration_points(pts) is True

    # Center (100, 100) should transform near (0, 0)
    cx, cy = calib.transform_pixel_to_canonical(100.0, 100.0)
    assert abs(cx) < 1e-4 and abs(cy) < 1e-4

    score = calib.get_score_for_pixel(100.0, 100.0)
    assert score.name == "BULL"

def test_dart_detector():
    calib = CalibrationEngine()
    pts = [(100.0, 20.0), (180.0, 100.0), (100.0, 180.0), (20.0, 100.0)]
    calib.set_calibration_points(pts)

    detector = DartDetector(calib)

    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 100
    img2 = img1.copy()
    # Draw a simulated dart hit near T20 (100, 60)
    cv2.circle(img2, (100, 60), 5, (255, 255, 255), -1)

    result = detector.detect_dart_impact(img1, img2)
    assert result is not None
    px, py, score = result
    assert score.sector == 20

def test_x01_game_logic():
    game = X01Game(["Alice", "Bob"], target_score=501, double_out=True)
    assert game.current_player.name == "Alice"
    assert game.current_player.current_score == 501

    # Alice throws 180
    for _ in range(3):
        res = game.throw_dart(ScoreResult("T20", 3, 20, 60))

    assert res["status"] == "turn_ended"
    assert game.players[0].current_score == 321
    assert game.players[0].average_3dart == 180.0
    assert game.current_player.name == "Bob"

    # Bob turns
    game.throw_dart(ScoreResult("S20", 1, 20, 20))
    assert game.players[1].current_score == 481

    # Test undo
    assert game.undo_last_dart() is True
    assert game.current_player.name == "Bob"
    assert game.players[1].current_score == 501
