FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    DIFFUSERS_NO_FLASH_ATTN=1 \
    TRANSFORMERS_ATTN_IMPLEMENTATION=eager

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Uninstall any flash-attn variants that may ship in the base image.
# They trigger torch.library registration crashes on some torch builds.
RUN pip uninstall -y flash-attn flash_attn flash-attn-3 flash_attn_3 2>/dev/null || true

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
