import argparse
import asyncio
import base64
import json
import logging
import time

import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedError, InvalidURI

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("yolo-client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

async def send_and_receive(ws_url: str, camera_index: int):
    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                logger.info(f"Connected to {ws_url}")
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    logger.error(f"Cannot open camera #{camera_index}")
                    return

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to grab frame")
                        break

                    # Encode and send
                    _, buf = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(buf).decode("utf-8")
                    await ws.send(json.dumps({"frame": b64}))

                    # Receive and display
                    msg = await ws.recv()
                    data = json.loads(msg)
                    ann_bytes = base64.b64decode(data["annotated_frame"])
                    nparr = np.frombuffer(ann_bytes, np.uint8)
                    annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    cv2.imshow("YOLO Annotated", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        except (ConnectionClosedError, InvalidURI) as e:
            logger.error(f"Connection error: {e}. Retrying in 5s…")
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Interrupted by user; exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="YOLO WebSocket Client")
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:8000/ws",
        help="WebSocket endpoint (e.g. ws://host:port/ws)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default: 0)"
    )
    args = parser.parse_args()

    try:
        asyncio.run(send_and_receive(args.ws_url, args.camera))
    except KeyboardInterrupt:
        logger.info("Client stopped")

if __name__ == "__main__":
    main()
