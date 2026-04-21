FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Python deps — layer caches well
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Handler
COPY handler.py .

# HF_TOKEN is provided at runtime via RunPod env vars.
# FLUX weights download on first request (cached across warm workers).
# This keeps the build fast and reliable — no 24GB download during docker build.

CMD ["python", "-u", "handler.py"]
