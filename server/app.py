import asyncio
import cv2
import base64
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from vision import CalibrationEngine, DartDetector, ScoreResult, auto_detect_dartboard_points
from game import X01Game, AIBot, CricketGame, AroundTheClockGame
from game.database import init_db, save_match, log_dart, log_vision_diagnostic
from analytics import AnalyticsEngine
import time

init_db()

app = FastAPI(title="Winmau Autodarts Lens Clone API")

# Mount static web directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances (in-memory for demo / single match)
calibration_engine = CalibrationEngine()
dart_detector = DartDetector(calibration_engine)
active_game: Optional[X01Game] = None
latest_frame_before: Optional[np.ndarray] = None

class NewGameRequest(BaseModel):
    game_mode: str = "x01"  # "x01", "cricket", "around_the_clock"
    players: List[str]
    target_score: int = 501
    double_out: bool = True
    include_bot: bool = False
    bot_skill: str = "intermediate"

ai_bot_instance: Optional[AIBot] = None
active_cricket_game: Optional[CricketGame] = None
active_atc_game: Optional[AroundTheClockGame] = None
active_game_type: str = "x01"

class CalibrationPointsRequest(BaseModel):
    points: List[List[float]]  # 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

class ManualThrowRequest(BaseModel):
    name: str
    multiplier: int
    sector: int
    value: int

@app.get("/")
def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/game/new")
def new_game(req: NewGameRequest):
    global active_game, active_cricket_game, active_atc_game, active_game_type, current_match_id, ai_bot_instance
    if not req.players:
        raise HTTPException(status_code=400, detail="Player list cannot be empty")

    player_list = list(req.players)
    if req.include_bot:
        ai_bot_instance = AIBot("Autodarts Bot", req.bot_skill)
        if ai_bot_instance.name not in player_list:
            player_list.append(ai_bot_instance.name)
    else:
        ai_bot_instance = None

    active_game_type = req.game_mode.lower()

    if active_game_type == "cricket":
        active_cricket_game = CricketGame(player_list)
        active_game = None
        active_atc_game = None
        return active_cricket_game.get_state()
    elif active_game_type == "around_the_clock":
        active_atc_game = AroundTheClockGame(player_list)
        active_game = None
        active_cricket_game = None
        return active_atc_game.get_state()
    else:
        active_game_type = "x01"
        active_game = X01Game(player_list, req.target_score, req.double_out)
        active_cricket_game = None
        active_atc_game = None
        current_match_id = save_match(req.target_score, req.double_out, None, player_list)
        return active_game.get_state()

@app.post("/api/calibration/auto")
async def auto_calibration(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    pts = auto_detect_dartboard_points(img)
    if pts is None:
        raise HTTPException(status_code=400, detail="Could not auto-detect dartboard")
    success = calibration_engine.set_calibration_points(pts)
    return {"calibrated": success, "points": pts}

@app.get("/api/analytics/stats")
def get_analytics_stats(player_name: Optional[str] = None):
    return AnalyticsEngine.get_player_summary_stats(player_name)

@app.get("/api/analytics/heatmap")
def get_analytics_heatmap(player_name: Optional[str] = None):
    return AnalyticsEngine.get_heatmap_data(player_name)

@app.get("/api/analytics/diagnostics")
def get_analytics_diagnostics():
    return AnalyticsEngine.get_vision_diagnostics()

@app.get("/api/game/state")
def get_game_state():
    if active_game_type == "cricket" and active_cricket_game is not None:
        return {"active": True, "game_type": "cricket", "state": active_cricket_game.get_state()}
    elif active_game_type == "around_the_clock" and active_atc_game is not None:
        return {"active": True, "game_type": "around_the_clock", "state": active_atc_game.get_state()}
    elif active_game is not None:
        return {"active": True, "game_type": "x01", "state": active_game.get_state()}
    return {"active": False}

@app.post("/api/game/throw")
def manual_throw(req: ManualThrowRequest):
    global active_game, active_cricket_game, active_atc_game, current_match_id
    score = ScoreResult(req.name, req.multiplier, req.sector, req.value)

    if active_game_type == "cricket" and active_cricket_game is not None:
        result = active_cricket_game.throw_dart(score)
        return {"result": result, "state": active_cricket_game.get_state()}
    elif active_game_type == "around_the_clock" and active_atc_game is not None:
        result = active_atc_game.throw_dart(score)
        return {"result": result, "state": active_atc_game.get_state()}
    elif active_game is not None:
        cx, cy = 0.0, 0.606 if req.sector == 20 else 0.0
        player = active_game.current_player
        result = active_game.throw_dart(score)

        if current_match_id is None:
            current_match_id = 1

        log_dart(
            match_id=current_match_id,
            player_name=player.name,
            leg_number=1,
            turn_number=(player.darts_thrown // 3) + 1,
            dart_number_in_turn=(player.darts_thrown % 3) or 3,
            dart_score_name=score.name,
            multiplier=score.multiplier,
            sector=score.sector,
            score_value=score.value,
            canonical_x=cx,
            canonical_y=cy,
            is_double_attempt=(player.current_score <= 40 and player.current_score % 2 == 0),
            is_checkout_hit=(player.current_score == 0)
        )
        return {"result": result, "state": active_game.get_state()}
    else:
        raise HTTPException(status_code=400, detail="No active game")

@app.post("/api/game/undo")
def undo_throw():
    global active_game
    if active_game is None:
        raise HTTPException(status_code=400, detail="No active game")
    success = active_game.undo_last_dart()
    return {"success": success, "state": active_game.get_state()}

@app.post("/api/calibration/set")
def set_calibration(req: CalibrationPointsRequest):
    if len(req.points) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 calibration points required [Top, Right, Bottom, Left]")
    pts = [(float(p[0]), float(p[1])) for p in req.points]
    success = calibration_engine.set_calibration_points(pts)
    return {"calibrated": success}

@app.post("/api/vision/process_images")
async def process_images(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...)
):
    """API Endpoint to test vision scoring by uploading before & after throw images."""
    if calibration_engine.H is None:
        raise HTTPException(status_code=400, detail="Dartboard must be calibrated first")

    before_bytes = await before_file.read()
    after_bytes = await after_file.read()

    np_before = np.frombuffer(before_bytes, np.uint8)
    np_after = np.frombuffer(after_bytes, np.uint8)

    img_before = cv2.imdecode(np_before, cv2.IMREAD_COLOR)
    img_after = cv2.imdecode(np_after, cv2.IMREAD_COLOR)

    if img_before is None or img_after is None:
        raise HTTPException(status_code=400, detail="Invalid image file(s)")

    detected = dart_detector.detect_dart_impact(img_before, img_after)
    if detected is None:
        return {"detected": False, "message": "No new dart impact detected"}

    tip_px, tip_py, score = detected

    # Automatically register throw if game is active
    game_res = None
    if active_game is not None:
        player = active_game.current_player
        game_res = active_game.throw_dart(score)
        if current_match_id is None:
            current_match_id = 1
        log_dart(
            match_id=current_match_id,
            player_name=player.name,
            leg_number=1,
            turn_number=(player.darts_thrown // 3) + 1,
            dart_number_in_turn=(player.darts_thrown % 3) or 3,
            dart_score_name=score.name,
            multiplier=score.multiplier,
            sector=score.sector,
            score_value=score.value,
            canonical_x=cx,
            canonical_y=cy,
            is_double_attempt=(player.current_score <= 40 and player.current_score % 2 == 0),
            is_checkout_hit=(player.current_score == 0),
            detection_time_ms=duration_ms
        )

    return {
        "detected": True,
        "tip_pixel": [tip_px, tip_py],
        "score": score.to_dict(),
        "game_result": game_res,
        "game_state": active_game.get_state() if active_game else None
    }

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "calibrate":
                points = data.get("points", [])
                if len(points) == 4:
                    pts = [(float(p[0]), float(p[1])) for p in points]
                    calibration_engine.set_calibration_points(pts)
                    await websocket.send_json({"type": "calibration_updated", "calibrated": True})

            elif msg_type == "frame_before":
                # Receive camera frame snapshot before throw
                global latest_frame_before
                img_data = data.get("image")
                if img_data:
                    encoded_data = img_data.split(',')[1] if ',' in img_data else img_data
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    latest_frame_before = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            elif msg_type == "frame_after":
                # Process throw frame
                img_data = data.get("image")
                if img_data and latest_frame_before is not None and calibration_engine.H is not None:
                    encoded_data = img_data.split(',')[1] if ',' in img_data else img_data
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    frame_after = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    detected = dart_detector.detect_dart_impact(latest_frame_before, frame_after)
                    if detected:
                        tip_px, tip_py, score = detected
                        latest_frame_before = frame_after.copy()

                        game_res = None
                        if active_game:
                            game_res = active_game.throw_dart(score)

                        await manager.broadcast({
                            "type": "dart_detected",
                            "score": score.to_dict(),
                            "tip": [tip_px, tip_py],
                            "game_result": game_res,
                            "game_state": active_game.get_state() if active_game else None
                        })
                    else:
                        await websocket.send_json({"type": "no_dart_detected"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
