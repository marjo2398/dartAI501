# Dart Vision Scoring System (Autodarts Lens Equivalent)

A web-based computer vision system for real-time automatic dart scoring and X01 game management using a standard camera (e.g., webcam) or image/video uploads.

## Features
- **Computer Vision Scoring Engine**: Homography-based board perspective calibration and differential image/motion dart tip detection.
- **X01 Game Engine**: Support for 501, 301, and custom X01 dart games with turn tracking, bust rules, double-out checkouts, and stats (3-dart averages).
- **Web UI**: Interactive dashboard with real-time video feed / image upload, interactive dartboard score calibration, and live match stats.
- **REST & WebSocket API**: Fast live state sync between vision detector, server, and client frontends.

## Structure
- `vision/`: Dartboard geometry, perspective calibration (homography), and dart impact detection algorithms.
- `game/`: X01 game rules engine, player score tracker, and history state management.
- `server/`: FastAPI server handling WebSockets, REST APIs, and vision pipeline integration.
- `static/`: HTML5/JS/CSS interactive frontend.
- `tests/`: Automated unit and integration tests.
