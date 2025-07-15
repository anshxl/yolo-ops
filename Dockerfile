FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# export to onnx runtime
RUN yolo export model=yolov8n.pt format=onnx imgsz=320

# quantize to INT8 dynamically
RUN python - <<EOF
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input="yolov8n.onnx",
    model_output="yolov8n_int8.onnx",
    weight_type=QuantType.QInt8
)
EOF

# Copy application code
COPY . .

# Tell Docker (and Render) that the container listens on $PORT
EXPOSE 80

# Exec form, but launches a shell so $PORT is expanded
ENTRYPOINT ["sh","-c","exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}"]
