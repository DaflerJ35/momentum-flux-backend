FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Python deps (layer caches well)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Handler
COPY handler.py .

# Pre-download FLUX.1-dev at build time so cold starts are fast.
# Requires HF_TOKEN build-arg because FLUX.1-dev is gated.
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN if [ -n "$HF_TOKEN" ]; then \
      python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN']); \
from diffusers import FluxPipeline; import torch; \
FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)" ; \
    fi

CMD ["python", "-u", "handler.py"]
