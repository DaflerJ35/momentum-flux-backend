FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps. Do NOT reinstall torch — the base image has the CUDA-enabled version.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY handler.py .

# HF_TOKEN is provided by RunPod at runtime.
# FLUX weights download on the first request (lazy load in handler).

CMD ["python", "-u", "handler.py"]
