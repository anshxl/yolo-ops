from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import base64, cv2, numpy as np
import logging
import os

# Configure logging
logger = logging.getLogger("uvicorn.error")

# Get environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")  # default to tiny COCO model
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))  # default confidence threshold

# Initialize FastAPI app
app = FastAPI()
model = YOLO("yolov8n.pt")  # latest tiny COCO model

# Utility function for testing
def annotate_frame(img: np.ndarray) -> np.ndarray:
    """
    Run YOLO inference on an OpenCV image and return an annotated image.
    Useful for unit‐testing this step in isolation.
    """
    results = model(img, conf=CONF_THRESH)[0]
    return results.plot()

# Health check endpoint
@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

# WebSocket endpoint for real-time inference
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WS connection accepted")
    try:
        while True:
            data = await ws.receive_json()
            # decode incoming frame
            img_bytes = base64.b64decode(data["frame"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # run inference & draw boxes
            results = model(img)[0]  # single frame
            annotated = results.plot()  # OpenCV image with boxes

            # encode back to JPEG + base64
            _, buf = cv2.imencode(".jpg", annotated)
            out_b64 = base64.b64encode(buf).decode("utf-8")

            # send back
            await ws.send_json({"annotated_frame": out_b64})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"Error during WS handling: {e}")
    finally:
        logger.info("WebSocket connection closed")

