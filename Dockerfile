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

# Copy application code
COPY . .

# Tell Docker (and Render) that the container listens on $PORT
EXPOSE 80

# Exec form, but launches a shell so $PORT is expanded
ENTRYPOINT [
  "sh", "-c",
  "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}"
]