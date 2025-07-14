# tests/test_ws.py

import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app, annotate_frame

client = TestClient(app)


def make_dummy_jpg_b64(width=128, height=128):
    """
    Create a plain blank image, encode to JPEG, then to base64 string.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # draw a white square so YOLO has something to “see”
    cv2.rectangle(img, (32, 32), (96, 96), (255, 255, 255), -1)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8"), img.shape


def test_healthz():
    """GET /healthz returns status OK."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_annotate_frame_function():
    """
    The annotate_frame utility should accept a numpy image
    and return an annotated numpy image of the same shape.
    """
    # create a dummy array
    dummy = np.zeros((100, 200, 3), dtype=np.uint8)
    out = annotate_frame(dummy)
    assert isinstance(out, np.ndarray)
    assert out.shape == dummy.shape


def test_ws_endpoint_annotations():
    """
    Connect to /ws, send one dummy frame, and verify
    the response contains a decodable JPEG of the same dimensions.
    """
    jpg_b64, (h, w, _) = make_dummy_jpg_b64()
    with client.websocket_connect("/ws") as ws:
        # send the dummy frame
        ws.send_json({"frame": jpg_b64})

        # receive annotated frame
        data = ws.receive_json()
        assert "annotated_frame" in data

        # decode and check image
        ann_bytes = base64.b64decode(data["annotated_frame"])
        arr = np.frombuffer(ann_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        # same width/height
        assert img.shape[0] == h and img.shape[1] == w
