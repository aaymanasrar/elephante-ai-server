# Elephante AI Server

FastAPI service that powers Elephante's on-device fashion intelligence: outfit
compatibility scoring (ResNet50 / FashionCLIP embeddings + Polyvore-trained
match head), SAM garment segmentation, and wardrobe-item analysis for the
Stylist's Vision feature.

Deployed on Render and consumed by the Elephante Next.js app via
`NEXT_PUBLIC_AI_MODEL_URL` (default `https://elephante-ai-server.onrender.com`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Liveness check (use as Render health check path) |
| GET | `/health` | Model + SAM status report |
| POST | `/analyze` | Wardrobe-item analysis for Stylist's Vision (called by `services/analyzeWardrobeWithAI.ts`) |
| POST | `/predict-match` | Shirt + pants compatibility score (multipart: `shirt_image`, `pants_image`, `skin_tone`, `use_segmentation`) |
| POST | `/segment-garment` | SAM garment segmentation (multipart: `image`, optional `box`, `point_coords`, `point_labels`, `crop`, `require_segmentation`) |
| GET | `/sam-status` | SAM availability details |
| POST | `/v1/chat/completions` (`/openai/v1/chat/completions`) | OpenAI-compatible chat completions, routed to Groq / local HF / Meta LLaMA API |

### POST /analyze

Accepts either contract — an uploaded image (`image_base64`) or the Elephante
web app's `image_url` + metadata form (`services/analyzeWardrobeWithAI.ts`).

Request (JSON, either form):

```json
{
  "image_url": "https://.../garment.jpg",
  "item_name": "Linen thobe",
  "item_type": "thobe",
  "color": "white",
  "occasion": "eid"
}
```

```json
{
  "image_base64": "data:image/png;base64,...",
  "item_type": "shirt"
}
```

When an image is available, it's sent to Groq's vision model
(`GROQ_MODEL`, default `meta-llama/llama-4-scout-17b-16e-instruct`) for
category/region/color/material/description using Saudi dress-code regions
(Najdi, Hijazi, Global MENA). Response:

```json
{
  "category": "thobe",
  "region": "Hijazi",
  "color": "white",
  "material": "linen",
  "description": "A lightweight white linen thobe suited to warm coastal climates.",
  "perfect_visual_prompt": "Professional fashion photography of a white Linen thobe, ...",
  "cultural_tags": ["Hijazi", "thobe", "linen"],
  "cultural_context": "A lightweight white linen thobe suited to warm coastal climates.",
  "style_origin": "Hijazi",
  "occasion_arabic": "العيد",
  "formality_level": "festive",
  "analysis_source": "groq",
  "status": "success"
}
```

If `GROQ_API_KEY` isn't configured, the image can't be fetched/decoded, or
Groq fails, the endpoint degrades to a heuristic (metadata-only) response
instead of failing — check `analysis_source` (`"groq"` vs `"heuristic"`) to
see which path served the request. Malformed `image_base64` still returns
`400`.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | Required for real (non-heuristic) `/analyze` results. `/health` reports `groq_configured`. |
| `GROQ_API_URL` | `https://api.groq.com` | Groq API base URL. |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision-capable Groq model used by `/analyze` and the OpenAI-compatible chat endpoint. |
| `HF_LOCAL_INFERENCE` | `false` | Run the OpenAI-compatible chat endpoint against a local Hugging Face model instead of Groq. |
| `HF_MODEL_NAME` | `humain-ai/ALLaM-7B-Instruct-preview` | Local HF model id, used when `HF_LOCAL_INFERENCE=true`. |
| `HF_DISABLE_CUDA` | `false` | Force CPU for local HF inference. |
| `LLAMA_USE_API` / `LLAMA_API_KEY` / `LLAMA_API_URL` | `false` / — / `https://llama.developer.meta.com` | Optional Meta LLaMA API backend for the chat endpoint. |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated allowed origins. Set to your Vercel domain(s) in production — the browser calls this server directly. |
| `SAM_CHECKPOINT` / SAM env vars | — | See `SAM_INTEGRATION.md`. |

Copy `.env.template` to `.env` and fill in the keys you need locally.

## OpenAI-compatible chat endpoint

`POST /v1/chat/completions` (and `/openai/v1/chat/completions`) proxies to
Groq, a local HF model, or the Meta LLaMA API depending on env config and the
requested `model`. Streaming (`stream: true`) is not supported.

## Model checkpoints

Place next to `main.py`:

- `elephante_stylist_brain.pt` — trained ResNet50 encoder (see `POLYVORE_TRAINING.md`)
- `elephante_match_head.pt` — compatibility head; its `encoder_type` field selects ResNet50 vs FashionCLIP
- `sam_vit_b_01ec64.pth` — optional, enables SAM segmentation

The server boots without any of them: it falls back to an ImageNet ResNet50
encoder and cosine-similarity scoring, and reports what it loaded at `/health`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
