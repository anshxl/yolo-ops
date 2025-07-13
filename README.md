## End-to-End YOLO MLOps Pipeline

This project demonstrates how to productionize a YOLOv8 inference service using FastAPI, WebSockets, Docker, GitHub Actions, and free-tier Render.com hosting.

### Architecture Diagram
```mermaid
graph TD
  A[GitHub Push] --> B[GitHub Actions]
  B --> C[Unit Tests & Lint]
  C --> D[Docker Build & Push to GHCR]
  D --> E[Render.com (Auto-Deploy)]
  E --> F[FastAPI WebSocket /ws]
```

### Usage
# Start Python client to stream webcam:
python client.py --ws-url ws://<your-domain>/ws