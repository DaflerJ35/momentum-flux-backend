FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Keep the CUDA torch that ships with the base image — do not reinstall.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
