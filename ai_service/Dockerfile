# 3D DCE-MRI Deep Learning AI Microservice Dockerfile
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (for matplotlib and image I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (CPU-optimized PyTorch to minimize image footprint)
COPY ai_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source code, models, and checkpoints
COPY ai_service /app/ai_service
COPY src /app/src
COPY checkpoints /app/checkpoints
COPY data /app/data

EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "ai_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
