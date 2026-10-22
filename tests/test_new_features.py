import pytest
import numpy as np
import cv2
from vision.autocalib import auto_detect_dartboard_points
from game import AIBot, CricketGame, AroundTheClockGame
from vision.dartboard import ScoreResult

def test_auto_calibration():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 120, (255, 255, 255), 4)
    pts = auto_detect_dartboard_points(img)
    assert pts is not None
    assert len(pts) == 4

def test_ai_bot():
    bot = AIBot("TestBot", "pro")
    throw = bot.generate_throw(501)
    assert isinstance(throw, ScoreResult)
    assert throw.multiplier > 0

def test_cricket_game():
    game = CricketGame(["P1", "P2"])
    # P1 hits T20 three times to close sector 20
    game.throw_dart(ScoreResult("T20", 3, 20, 60))
    assert game.marks["P1"][20] == 3
    assert game.scores["P1"] == 0

    # P1 hits T20 again -> scores 60 points since sector 20 is closed by P1 but open for P2
    game.throw_dart(ScoreResult("T20", 3, 20, 60))
    assert game.scores["P1"] == 60

def test_around_the_clock_game():
    game = AroundTheClockGame(["P1"])
    assert game.targets["P1"] == 1

    # P1 hits S1
    game.throw_dart(ScoreResult("S1", 1, 1, 1))
    assert game.targets["P1"] == 2
