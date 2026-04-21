# Momentum AI Creator — RunPod Serverless Handler

FLUX.1-dev text-to-image handler for **Momentum AI Creator**.

- Text-to-image only. No face-reference uploads, no img2img from uploaded photos, no NSFW bypass.
- Character consistency is achieved via deterministic `seed` + detailed prompt text (set per-character in the Momentum frontend).
- Optional per-character LoRAs can be loaded from `/runpod-volume/loras/<name>.safetensors` (trained on synthetic data only).

## Files

- `handler.py` — the serverless handler
- `Dockerfile` — builds a prod image (pre-downloads FLUX.1-dev during build)
- `requirements.txt` — pinned Python deps

## Deploy via RunPod GitHub integration (easiest)

1. Push this repo to GitHub (public or private).
2. RunPod Console → **Serverless → New Endpoint → Source: GitHub**.
3. Connect your GitHub account, pick this repo and branch.
4. **Build Configuration**:
   - Dockerfile path: `backend/Dockerfile`
   - Build args: add `HF_TOKEN=hf_xxxxxxxx`
     (create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the FLUX.1-dev license at [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev))
5. **GPU**: A100 80GB or L40S (FLUX needs real VRAM).
6. **Container Disk**: 40 GB.
7. **Max Workers**: 2–3. **Idle Timeout**: 5s.
8. Create → copy the new **Endpoint ID** → paste in Momentum's Settings tab.

## Deploy locally with Docker (if you prefer)

```bash
docker login
docker build --build-arg HF_TOKEN=hf_xxx -t YOURNAME/momentum-flux:latest .
docker push YOURNAME/momentum-flux:latest
```
Then create the RunPod endpoint pointing at `YOURNAME/momentum-flux:latest`.

## API

### Request
```json
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "input": {
    "prompt": "a portrait of Ada Rivers, warm olive skin, auburn hair, confident",
    "negative_prompt": "blurry, deformed",
    "seed": 1234567890,
    "steps": 24,
    "guidance": 3.5,
    "width": 1024,
    "height": 1024,
    "batch_size": 1,
    "scheduler": "euler",
    "style_preset": "portrait",
    "enhance_prompt": true,
    "character_lora": "ada_rivers_v1",
    "character_lora_weight": 0.85,
    "upscale_factor": 1.0
  }
}
```

### Response
```json
{
  "status": "success",
  "images": [
    { "image": "<base64 PNG>", "seed": 1234567890, "index": 0 }
  ],
  "metadata": {
    "execution_time_seconds": 8.41,
    "dimensions": "1024x1024",
    "scheduler": "euler",
    "steps": 24,
    "guidance": 3.5,
    "prompt_used": "portrait photograph, <your prompt>, <realism suffix>",
    "base_seed": 1234567890
  }
}
```

## Style presets
`portrait` · `editorial` · `cinematic` · `concept_art` · `product` · `anime` · `raw`
