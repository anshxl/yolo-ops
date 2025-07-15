# client.py

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

logger = logging.getLogger("yolo-client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


async def send_and_receive(ws_url: str, camera_index: int):
    """
    Connect to ws_url, capture frames from camera_index,
    send each as base64-JPEG JSON, receive annotated frames,
    and display them in an OpenCV window.
    """
    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                logger.info(f"Connected to {ws_url}")
                cap = cv2.VideoCapture(camera_index)
                # create window for displaying annotated frames
                cv2.namedWindow("YOLO Annotated", cv2.WINDOW_NORMAL)

                if not cap.isOpened():
                    logger.error(f"Cannot open camera #{camera_index}")
                    return

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to grab frame")
                        break

                    _, buf = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(buf).decode("utf-8")
                    await ws.send(json.dumps({"frame": b64}))

                    msg = await ws.recv()
                    data = json.loads(msg)
                    ann_bytes = base64.b64decode(data["annotated_frame"])
                    nparr = np.frombuffer(ann_bytes, np.uint8)
                    annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    logger.debug("about to show frame")
                    cv2.imshow("YOLO Annotated", annotated)
                    logger.debug("called imshow(), waiting for key press")

                    # key handling
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        except (ConnectionClosedError, InvalidURI) as e:
            logger.error(f"Connection error: {e}. Retrying in 5s...")
            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Interrupted by user; exiting.")
            break

        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="YOLO WebSocket Client"
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:8000/ws",
        help="WebSocket endpoint (e.g. ws://host:port/ws)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(send_and_receive(args.ws_url, args.camera))
    except KeyboardInterrupt:
        logger.info("Client stopped")


if __name__ == "__main__":
    main()
