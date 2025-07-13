from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import base64, cv2, numpy as np

app = FastAPI()
model = YOLO("yolov8n.pt")  # latest tiny COCO model

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
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
        pass
