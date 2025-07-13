import asyncio, websockets, base64, cv2, json
import numpy as np

WS_URL = "ws://localhost:8000/ws"

async def send_and_receive():
    async with websockets.connect(WS_URL) as ws:
        cap = cv2.VideoCapture(0)  # webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # encode frame to JPEG, then to base64
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf).decode("utf-8")
            await ws.send(json.dumps({"frame": b64}))

            # get annotated frame
            msg = await ws.recv()
            data = json.loads(msg)
            ann_bytes = base64.b64decode(data["annotated_frame"])
            nparr = np.frombuffer(ann_bytes, np.uint8)
            annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # display
            cv2.imshow("YOLO Annotated", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(send_and_receive())
