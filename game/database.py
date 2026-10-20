import sqlite3
import os
import json
from typing import List, Dict, Optional, Tuple

DB_PATH = "darts_analytics.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Matches table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        target_score INTEGER,
        double_out BOOLEAN,
        winner_name TEXT,
        players_json TEXT
    )
    """)

    # Darts thrown table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS darts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        player_name TEXT,
        leg_number INTEGER,
        turn_number INTEGER,
        dart_number_in_turn INTEGER,
        dart_score_name TEXT,
        multiplier INTEGER,
        sector INTEGER,
        score_value INTEGER,
        canonical_x REAL,
        canonical_y REAL,
        target_sector INTEGER,
        is_double_attempt BOOLEAN,
        is_checkout_hit BOOLEAN,
        detection_time_ms REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (match_id) REFERENCES matches(id)
    )
    """)

    # Vision diagnostics log table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vision_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        detection_duration_ms REAL,
        diff_mean_intensity REAL,
        detected_dart BOOLEAN,
        canonical_x REAL,
        canonical_y REAL
    )
    """)

    conn.commit()
    conn.close()

def save_match(target_score: int, double_out: bool, winner_name: Optional[str], players: List[str]) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO matches (target_score, double_out, winner_name, players_json) VALUES (?, ?, ?, ?)",
        (target_score, double_out, winner_name, json.dumps(players))
    )
    match_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return match_id

def log_dart(
    match_id: int,
    player_name: str,
    leg_number: int,
    turn_number: int,
    dart_number_in_turn: int,
    dart_score_name: str,
    multiplier: int,
    sector: int,
    score_value: int,
    canonical_x: float,
    canonical_y: float,
    target_sector: Optional[int] = None,
    is_double_attempt: bool = False,
    is_checkout_hit: bool = False,
    detection_time_ms: float = 0.0
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO darts (
            match_id, player_name, leg_number, turn_number, dart_number_in_turn,
            dart_score_name, multiplier, sector, score_value, canonical_x, canonical_y,
            target_sector, is_double_attempt, is_checkout_hit, detection_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        match_id, player_name, leg_number, turn_number, dart_number_in_turn,
        dart_score_name, multiplier, sector, score_value, canonical_x, canonical_y,
        target_sector, is_double_attempt, is_checkout_hit, detection_time_ms
    ))
    conn.commit()
    conn.close()

def log_vision_diagnostic(duration_ms: float, diff_intensity: float, detected: bool, x: float = 0.0, y: float = 0.0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO vision_logs (detection_duration_ms, diff_mean_intensity, detected_dart, canonical_x, canonical_y) VALUES (?, ?, ?, ?, ?)",
        (duration_ms, diff_intensity, detected, x, y)
    )
    conn.commit()
    conn.close()
