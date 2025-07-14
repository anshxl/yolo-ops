# main.py

import os
import logging
import base64

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))

app = FastAPI()
model = YOLO(MODEL_PATH)


def annotate_frame(img: np.ndarray) -> np.ndarray:
    """
    Run YOLO inference on an OpenCV image and return
    an annotated image.
    """
    results = model(img, conf=CONF_THRESH)[0]
    return results.plot()


@app.get("/")
async def root():
    return {"status": "live"}


@app.get("/healthz")
async def healthz():
    """
    Health-check endpoint.
    """
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    WebSocket endpoint expecting JSON:
      {"frame": "<base64-JPEG>"}
    and returning:
      {"annotated_frame": "<base64-JPEG>"}
    """
    await ws.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            payload = await ws.receive_json()
            img_bytes = base64.b64decode(payload["frame"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            annotated = annotate_frame(img)

            _, buf = cv2.imencode(".jpg", annotated)
            out_b64 = base64.b64encode(buf).decode("utf-8")

            await ws.send_json({"annotated_frame": out_b64})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")

    except Exception as e:
        logger.error(f"Error during WS handling: {e}")

    finally:
        logger.info("WebSocket connection closed")
