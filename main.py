import base64
import binascii
import colorsys
import io
import ipaddress
import json
import logging
import os
import socket
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from torchvision import transforms

from model_components import (
    CompatibilityHead,
    build_stylist_encoder,
    build_fashionclip_encoder,
    CLIP_MEAN,
    CLIP_STD,
)
from sam_segmentation import (
    SamSegmenter,
    SamUnavailableError,
    apply_mask,
    mask_quality,
    mask_to_png_bytes,
)


logger = logging.getLogger("elephante-ai")

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com")
# llama-3.2-11b-vision-preview was decommissioned by Groq; Llama 4 Scout is
# the current vision-capable model. Override via GROQ_MODEL env var.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

HF_LOCAL_INFERENCE = os.environ.get("HF_LOCAL_INFERENCE", "false").strip().lower() in ("1", "true", "yes")
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "humain-ai/ALLaM-7B-Instruct-preview")
HF_DISABLE_CUDA = os.environ.get("HF_DISABLE_CUDA", "false").strip().lower() in ("1", "true", "yes")
HF_DEVICE = "cuda" if torch.cuda.is_available() and not HF_DISABLE_CUDA else "cpu"

LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "")
LLAMA_API_URL = os.environ.get("LLAMA_API_URL", "https://llama.developer.meta.com")
LLAMA_USE_API = os.environ.get("LLAMA_USE_API", "false").strip().lower() in ("1", "true", "yes")

_hf_model = None
_hf_tokenizer = None
_hf_lock = threading.Lock()

app = FastAPI(title="Elephante AI Stylist API")

# The Elephante web app calls this server directly from the browser
# (NEXT_PUBLIC_AI_MODEL_URL), so CORS must be enabled. Restrict origins in
# production via CORS_ALLOW_ORIGINS (comma-separated), e.g.
#   CORS_ALLOW_ORIGINS=https://elephante.vercel.app,http://localhost:3000
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).with_name("elephante_stylist_brain.pt")
MATCH_HEAD_PATH = Path(__file__).with_name("elephante_match_head.pt")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_SIZE = 224

device = torch.device("cpu")


def _encoder_type_from_checkpoint(path: Path) -> str:
    """Read encoder_type stored in the match head checkpoint, default to resnet50."""
    if not path.exists():
        return "resnet50"
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("encoder_type", "resnet50")
    except Exception:
        return "resnet50"


def load_match_head(path: Path):
    if not path.exists():
        print("No match head found. Using cosine similarity fallback.")
        return None

    checkpoint = torch.load(path, map_location=device)
    head_kwargs = checkpoint.get("head_kwargs", {})
    state_dict = checkpoint.get("state_dict", checkpoint)
    head = CompatibilityHead(**head_kwargs)
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    encoder_type = checkpoint.get("encoder_type", "resnet50")
    print(f"Match head loaded  (encoder={encoder_type})")
    return head


# 1. Detect which encoder the match head was trained with, then load it.
encoder_type = _encoder_type_from_checkpoint(MATCH_HEAD_PATH)

if encoder_type == "fashionclip":
    print("Loading FashionCLIP encoder (Marqo/marqo-fashionCLIP)…")
    model = build_fashionclip_encoder()
    model.to(device)
    model.eval()
    print("FashionCLIP encoder ready.")
    digital_tailor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
else:
    print("Loading Elephante Brain (ResNet50)…")
    model = build_stylist_encoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Brain loaded successfully!")
    # 3. Setup the Digital Tailor.
    digital_tailor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
    ])

# 2. Load the compatibility head.
match_head = load_match_head(MATCH_HEAD_PATH)
sam_segmenter = SamSegmenter.from_env(device=device)


def undertone_from_skin_tone(skin_tone: str) -> str:
    skin_tone = (skin_tone or "").lower()
    if "warm" in skin_tone:
        return "warm"
    if "cool" in skin_tone:
        return "cool"
    return "neutral"


async def read_upload_image(upload: UploadFile, field_name: str) -> Image.Image:
    image_data = await upload.read()

    if not image_data:
        raise ValueError(f"{field_name} is empty.")

    if len(image_data) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES // (1024 * 1024)
        raise ValueError(f"{field_name} is too large. Max size is {max_mb} MB.")

    try:
        image = Image.open(io.BytesIO(image_data))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{field_name} must be a valid image.") from exc


def parse_base64_image(image_base64: str) -> Image.Image:
    if not image_base64:
        raise ValueError("image_base64 is required.")

    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 must be valid base64 encoded image data.") from exc

    if len(image_data) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES // (1024 * 1024)
        raise ValueError(f"image_base64 is too large. Max size is {max_mb} MB.")

    try:
        image = Image.open(io.BytesIO(image_data))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image_base64 must be a valid image.") from exc


def load_local_hf_model():
    global _hf_model, _hf_tokenizer

    with _hf_lock:
        if _hf_model is not None and _hf_tokenizer is not None:
            return _hf_model, _hf_tokenizer

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for local HF inference. Install it and retry."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if HF_DEVICE == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map="auto",
                **model_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map={"": "cpu"},
                **model_kwargs,
            )

        _hf_tokenizer = tokenizer
        _hf_model = model
        return _hf_model, _hf_tokenizer


def local_hf_chat_completion(messages: List[Dict], temperature: float, max_new_tokens: int) -> str:
    model, tokenizer = load_local_hf_model()

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        prompt_pairs = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_pairs.append(f"<{role}>: {content}")
        prompt_text = "\n".join(prompt_pairs)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        return_token_type_ids=False,
        padding=True,
    )

    if HF_DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 50,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def build_llama_api_client():
    try:
        from llama_api_client import LlamaAPIClient
    except ImportError as exc:
        raise RuntimeError(
            "The 'llama-api-client' package is required for Meta LLaMA API support. Install it and retry."
        ) from exc

    if not LLAMA_API_KEY:
        raise RuntimeError("LLAMA_API_KEY is required for Meta LLaMA API support.")

    return LlamaAPIClient(api_key=LLAMA_API_KEY, base_url=LLAMA_API_URL)


def llama_api_chat_completion(messages: List[Dict], model: str, temperature: float, max_tokens: int) -> str:
    client = build_llama_api_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if hasattr(response, "completion_message"):
        return response.completion_message

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and choice.message:
            return getattr(choice.message, "content", str(choice))
        return str(choice)

    return str(response)


def image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=85)
    return "data:image/{};base64,{}".format(fmt.lower(), base64.b64encode(buffer.getvalue()).decode("ascii"))


def build_groq_client(api_key: Optional[str] = None):
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Groq SDK is required for /analyze and OpenAI-compatible endpoints. Install 'groq' and set GROQ_API_KEY."
        ) from exc

    chosen_key = api_key or GROQ_API_KEY
    if not chosen_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    kwargs = {"api_key": chosen_key}
    if GROQ_API_URL:
        kwargs["base_url"] = GROQ_API_URL
    return Groq(**kwargs)


def get_authorization_api_key(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None

    parts = auth.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    return parts[1].strip()


def extract_response_text(response) -> str:
    # groq / OpenAI chat.completions shape
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if content:
            return content
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "output"):
        output = response.output
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, list) and content:
                    item = content[0]
                    return item.get("text") or item.get("content") or str(item)
                return str(first)
            return str(first)
    return str(response)


def openai_messages_to_groq_input(messages: List[Dict]) -> List[Dict]:
    result = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "")
        if role in {"system", "user", "assistant"}:
            result.append({"role": role, "content": content})
        else:
            result.append({"role": "user", "content": content})
    return result


def build_openai_chat_response(raw_text: str, model: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": raw_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }


def extract_json_object(text: str) -> dict:
    if not text or not isinstance(text, str):
        raise ValueError("Unable to parse empty response from Groq.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Groq response did not contain valid JSON.")
        return json.loads(text[start : end + 1])


def normalize_analyze_result(parsed: dict) -> dict:
    return {
        "category": str(parsed.get("category", "")).strip(),
        "region": str(parsed.get("region", "")).strip(),
        "color": str(parsed.get("color", "")).strip(),
        "material": str(parsed.get("material", "")).strip(),
        "description": str(parsed.get("description", "")).strip(),
        "perfect_visual_prompt": str(parsed.get("perfect_visual_prompt", "")).strip(),
    }


def analyze_with_groq(image: Image.Image) -> dict:
    client = build_groq_client()
    image_url = image_to_data_url(image, "JPEG")

    system_prompt = (
        "You are a Saudi and Greater MENA fashion analyst. Identify the garment category, region, dominant color, material, a brief description, "
        "and a perfect visual prompt for a design model. Classify the style using Saudi dress codes: Najdi for structured, formal, winter wool/heavy garments; "
        "Hijazi for lighter linen and coastal western Saudi styles with turbans/Amama; and Global MENA for contemporary regional or modern Arabic fashion. "
        "Respond only with a JSON object and do not add any extra explanation. "
        "The JSON must contain exactly these keys: category, region, color, material, description, perfect_visual_prompt."
    )

    user_prompt = (
        "Analyze the attached image and return only a single JSON object containing category, region, color, material, description, perfect_visual_prompt. "
        "Use the Saudi dress code categories Najdi, Hijazi, and Global MENA."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        max_tokens=500,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw_text = extract_response_text(response)
    parsed = extract_json_object(raw_text)
    return normalize_analyze_result(parsed)


def embed_image(image: Image.Image) -> torch.Tensor:
    """Average original + mirrored embeddings for a steadier clothing signal."""
    views = [
        digital_tailor(image),
        digital_tailor(ImageOps.mirror(image)),
    ]
    batch = torch.stack(views).to(device)

    with torch.inference_mode():
        embedding = model(batch).mean(dim=0, keepdim=True)
        return F.normalize(embedding, dim=1)


def score_visual_match(left_embedding: torch.Tensor, right_embedding: torch.Tensor) -> tuple:
    if match_head is not None:
        with torch.inference_mode():
            probability = torch.sigmoid(match_head(left_embedding, right_embedding)).item()
        return probability * 100, "polyvore_match_head"

    similarity = F.cosine_similarity(left_embedding, right_embedding)
    score = float(((similarity.item() + 1) / 2) * 100)
    return score, "cosine_similarity"


def average_rgb(pixels):
    count = len(pixels)
    if count == 0:
        return [0, 0, 0]

    red = sum(pixel[0] for pixel in pixels) / count
    green = sum(pixel[1] for pixel in pixels) / count
    blue = sum(pixel[2] for pixel in pixels) / count
    return [round(red), round(green), round(blue)]


def analyze_clothing_color(image: Image.Image) -> dict:
    small_image = image.resize((64, 64), Image.Resampling.BILINEAR)
    pixels = list(small_image.getdata())

    # Product photos often have white studio backgrounds. Keep saturated pixels
    # first, then fall back to all non-background pixels for neutral garments.
    non_background_pixels = []
    chromatic_pixels = []
    warm_score = 0.0
    cool_score = 0.0
    neutral_score = 0.0

    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        if v > 0.93 and s < 0.12:
            continue

        non_background_pixels.append((r, g, b))

        if s < 0.14 or v < 0.12:
            neutral_score += 1.0
            continue

        chromatic_pixels.append((r, g, b))
        hue_degrees = h * 360
        weight = s * max(v, 0.25)

        if hue_degrees <= 65 or hue_degrees >= 300:
            warm_score += weight
        else:
            cool_score += weight

    sampled_pixels = chromatic_pixels or non_background_pixels or pixels
    chromatic_score = warm_score + cool_score
    total_score = chromatic_score + neutral_score

    if total_score == 0:
        temperature = "neutral"
        confidence = 1.0
    elif chromatic_score < neutral_score * 0.8:
        temperature = "neutral"
        confidence = min(1.0, neutral_score / total_score)
    elif warm_score >= cool_score:
        temperature = "warm"
        confidence = warm_score / chromatic_score
    else:
        temperature = "cool"
        confidence = cool_score / chromatic_score

    return {
        "temperature": temperature,
        "confidence": round(float(confidence), 3),
        "average_rgb": average_rgb(sampled_pixels),
    }


def color_alignment_bonus(
    item_color: dict,
    target_undertone: str,
    match_bonus: float,
    clash_penalty: float,
) -> float:
    if target_undertone == "neutral" or item_color["temperature"] == "neutral":
        return 0.0

    confidence = item_color["confidence"]
    if item_color["temperature"] == target_undertone:
        return match_bonus * confidence
    return clash_penalty * confidence


def outfit_color_bonus(shirt_color: dict, pants_color: dict, user_undertone: str) -> float:
    bonus = 0.0
    bonus += color_alignment_bonus(
        shirt_color,
        user_undertone,
        match_bonus=8.0,
        clash_penalty=-4.0,
    )
    bonus += color_alignment_bonus(
        pants_color,
        user_undertone,
        match_bonus=4.0,
        clash_penalty=-2.0,
    )

    shirt_temp = shirt_color["temperature"]
    pants_temp = pants_color["temperature"]
    pair_confidence = min(shirt_color["confidence"], pants_color["confidence"])

    if shirt_temp == "neutral" or pants_temp == "neutral":
        bonus += 3.0
    elif shirt_temp == pants_temp:
        bonus += 4.0 * pair_confidence
    else:
        bonus -= 3.0 * pair_confidence

    return bonus


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def full_mask_png_base64(image: Image.Image) -> str:
    mask = Image.new("L", image.size, 255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def original_image_fallback_payload(image: Image.Image, reason: str) -> dict:
    width, height = image.size
    return {
        "score": None,
        "area": width * height,
        "area_ratio": 1.0,
        "bbox_xyxy": [0, 0, width, height],
        "bbox_xywh": [0, 0, width, height],
        "prompt_box_xyxy": None,
        "model_type": None,
        "checkpoint": None,
        "mask_png_base64": full_mask_png_base64(image),
        "isolated_png_base64": encode_png_base64(image),
        "segmentation_status": "fallback_original",
        "segmentation_reason": reason,
        "status": "success",
    }


def parse_box_prompt(box: Optional[str]):
    if not box:
        return None

    parts = [part.strip() for part in box.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("box must be formatted as x1,y1,x2,y2.")

    return [float(part) for part in parts]


def parse_point_prompts(point_coords: Optional[str], point_labels: Optional[str]):
    if not point_coords:
        return None, None

    coords = []
    for pair in point_coords.split(";"):
        pair = pair.strip()
        if not pair:
            continue

        values = [value.strip() for value in pair.split(",") if value.strip()]
        if len(values) != 2:
            raise ValueError("point_coords must be formatted as x,y;x,y.")
        coords.append([float(values[0]), float(values[1])])

    if not coords:
        raise ValueError("point_coords did not contain any valid points.")

    if not point_labels:
        return coords, [1] * len(coords)

    labels = [int(value.strip()) for value in point_labels.split(",") if value.strip()]
    if len(labels) != len(coords):
        raise ValueError("point_labels must contain one label per point.")

    invalid_labels = [label for label in labels if label not in (0, 1)]
    if invalid_labels:
        raise ValueError("point_labels must only contain 0 (background) or 1 (foreground).")

    return coords, labels


async def maybe_segment_clothing(
    image: Image.Image,
    field_name: str,
    enabled: bool,
):
    if not enabled:
        return image, {"status": "disabled"}

    try:
        result = await run_in_threadpool(sam_segmenter.segment, image)
        quality = mask_quality(result.mask)
        if not quality["usable"]:
            metadata = result.to_metadata()
            metadata.update(quality)
            metadata["status"] = "fallback_original"
            return image, metadata

        isolated = apply_mask(image, result.mask, crop=True)
        metadata = result.to_metadata()
        metadata.update(quality)
        metadata["status"] = "applied"
        return isolated, metadata
    except SamUnavailableError as exc:
        logger.info("SAM skipped for %s: %s", field_name, exc)
        return image, {"status": "skipped", "reason": str(exc)}
    except Exception as exc:
        logger.exception("SAM segmentation failed for %s", field_name)
        return image, {"status": "failed", "error": str(exc)}


@app.get("/sam-status")
async def sam_status():
    return JSONResponse(content=sam_segmenter.status())


@app.post("/segment-garment")
async def segment_garment(
    image: UploadFile = File(...),
    box: Optional[str] = Form(None),
    point_coords: Optional[str] = Form(None),
    point_labels: Optional[str] = Form(None),
    crop: bool = Form(True),
    require_segmentation: bool = Form(False),
):
    img = None
    try:
        img = await read_upload_image(image, "image")
        parsed_box = parse_box_prompt(box)
        parsed_points, parsed_labels = parse_point_prompts(point_coords, point_labels)
        result = await run_in_threadpool(
            sam_segmenter.segment,
            img,
            parsed_box,
            parsed_points,
            parsed_labels,
        )
        quality = mask_quality(result.mask)
        if not quality["usable"]:
            if require_segmentation:
                return JSONResponse(
                    content={"error": quality["reason"], "status": "failed"},
                    status_code=422,
                )

            metadata = result.to_metadata()
            payload = original_image_fallback_payload(img, quality["reason"])
            payload.update(metadata)
            payload.update(quality)
            payload["segmentation_status"] = "fallback_original"
            return JSONResponse(content=payload)

        isolated = apply_mask(img, result.mask, crop=crop)

        payload = result.to_metadata()
        payload.update(quality)
        payload.update(
            {
                "mask_png_base64": base64.b64encode(
                    mask_to_png_bytes(result.mask)
                ).decode("ascii"),
                "isolated_png_base64": encode_png_base64(isolated),
                "segmentation_status": "applied",
                "status": "success",
            }
        )
        return JSONResponse(content=payload)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=400)
    except SamUnavailableError as exc:
        if require_segmentation:
            return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=503)

        return JSONResponse(content=original_image_fallback_payload(img, str(exc)))
    except Exception as exc:
        logger.exception("SAM segmentation failed")
        if require_segmentation:
            return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=500)

        if img is None:
            return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=500)

        return JSONResponse(content=original_image_fallback_payload(img, str(exc)))


# 4. The API Endpoint Next.js will call.
@app.post("/predict-match")
async def predict_match(
    shirt_image: UploadFile = File(...),
    pants_image: UploadFile = File(...),
    skin_tone: str = Form("neutral"),
    use_segmentation: bool = Form(True),
):
    try:
        shirt_img = await read_upload_image(shirt_image, "shirt_image")
        pants_img = await read_upload_image(pants_image, "pants_image")
        shirt_img, shirt_segmentation = await maybe_segment_clothing(
            shirt_img,
            "shirt_image",
            use_segmentation,
        )
        pants_img, pants_segmentation = await maybe_segment_clothing(
            pants_img,
            "pants_image",
            use_segmentation,
        )

        shirt_color = analyze_clothing_color(shirt_img)
        pants_color = analyze_clothing_color(pants_img)
        user_undertone = undertone_from_skin_tone(skin_tone)

        shirt_barcode = embed_image(shirt_img)
        pants_barcode = embed_image(pants_img)

        model_score, model_source = score_visual_match(shirt_barcode, pants_barcode)

        color_adjustment = outfit_color_bonus(shirt_color, pants_color, user_undertone)
        final_score = max(0.0, min(100.0, model_score + color_adjustment))

        return JSONResponse(content={
            "match_score": round(final_score, 2),
            "model_score": round(model_score, 2),
            "model_source": model_source,
            "color_adjustment": round(color_adjustment, 2),
            "skin_tone_analyzed": skin_tone,
            "user_undertone": user_undertone,
            "shirt_detected_as": shirt_color["temperature"],
            "pants_detected_as": pants_color["temperature"],
            "shirt_color": shirt_color,
            "pants_color": pants_color,
            "segmentation": {
                "enabled": use_segmentation,
                "shirt": shirt_segmentation,
                "pants": pants_segmentation,
            },
            "status": "success",
        })
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=400)
    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=500)


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class AnalyzeRequest(BaseModel):
    """Accepts both contracts: raw base64 uploads AND the Elephante web app
    (services/analyzeWardrobeWithAI.ts), which sends image_url + metadata."""

    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    item_name: str = ""
    item_type: str = ""
    color: str = ""
    occasion: str = "casual"


def _assert_public_http_url(url: str) -> None:
    """Reject non-HTTP schemes and private/loopback hosts (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("image_url must use http or https.")
    if not parsed.hostname:
        raise ValueError("image_url is missing a hostname.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError("image_url hostname could not be resolved.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ValueError("image_url must point to a public host.")


def fetch_image_from_url(url: str) -> Image.Image:
    _assert_public_http_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "elephante-ai-server/1.1"})
    with urllib.request.urlopen(request, timeout=10) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image_url content exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image_url did not return a valid image.") from exc


OCCASION_ARABIC = {
    "casual": "كاجوال", "everyday": "يومي", "formal": "رسمي", "business": "عمل",
    "work": "عمل", "party": "حفلة", "wedding": "زفاف", "evening": "سهرة",
    "sport": "رياضي", "sports": "رياضي", "gym": "رياضي", "beach": "شاطئ",
    "travel": "سفر", "traditional": "تقليدي", "eid": "العيد", "ramadan": "رمضان",
    "date": "موعد", "brunch": "غداء",
}

OCCASION_FORMALITY = {
    "casual": "casual", "everyday": "casual", "beach": "casual", "travel": "casual",
    "gym": "athletic", "sport": "athletic", "sports": "athletic",
    "business": "business", "work": "business", "brunch": "smart casual",
    "date": "smart casual", "party": "semi-formal", "evening": "semi-formal",
    "formal": "formal", "wedding": "formal", "eid": "festive",
    "ramadan": "festive", "traditional": "traditional",
}


def heuristic_analysis(payload: AnalyzeRequest) -> dict:
    """Metadata-only fallback when Groq/LLM analysis is unavailable."""
    item_label = (payload.item_name or payload.item_type or "garment").strip()
    color_label = (payload.color or "").strip()
    descriptor = f"{color_label} {item_label}".strip()
    occasion_key = (payload.occasion or "casual").strip().lower()
    return {
        "category": payload.item_type or "",
        "region": "",
        "color": color_label,
        "material": "",
        "description": f"{descriptor} for a {occasion_key} occasion.",
        "perfect_visual_prompt": (
            f"Professional fashion photography of a {descriptor}, "
            f"styled for a {occasion_key} occasion, worn by a fashion model, "
            "clean neutral studio background, soft diffused lighting, "
            "ultra-detailed fabric texture, full garment in frame, editorial quality, 4k"
        ),
    }


def enrich_analysis_for_client(result: dict, payload: AnalyzeRequest, source: str) -> dict:
    """Add the fields the Elephante web client reads (analyzeWardrobeWithAI.ts)."""
    occasion_key = (payload.occasion or "casual").strip().lower()
    region = result.get("region", "")
    enriched = dict(result)
    enriched.update({
        "cultural_tags": [tag for tag in [region, result.get("category"), result.get("material")] if tag],
        "cultural_context": result.get("description", ""),
        "style_origin": region or "Contemporary global",
        "occasion_arabic": OCCASION_ARABIC.get(occasion_key, payload.occasion or "كاجوال"),
        "formality_level": OCCASION_FORMALITY.get(occasion_key, "smart casual"),
        "analysis_source": source,
        "status": "success",
    })
    return enriched


@app.post("/openai/v1/chat/completions")
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request, payload: OpenAIChatRequest):
    if payload.stream:
        return JSONResponse(
            content={"error": "Streaming is not supported by this endpoint.", "status": "failed"},
            status_code=400,
        )

    model = payload.model or (HF_MODEL_NAME if HF_LOCAL_INFERENCE else GROQ_MODEL)
    lower_model = model.lower()
    is_local = (
        HF_LOCAL_INFERENCE
        and (
            lower_model == HF_MODEL_NAME.lower()
            or lower_model in {
                "allam-7b",
                "allam-7b-instruct-preview",
                "humain-ai/allam-7b-instruct-preview",
            }
        )
    )

    if is_local:
        raw_text = await run_in_threadpool(
            local_hf_chat_completion,
            payload.messages,
            payload.temperature if payload.temperature is not None else 0.1,
            payload.max_tokens or 500,
        )
        payload_result = build_openai_chat_response(raw_text, model)
        return JSONResponse(content=payload_result)

    if LLAMA_USE_API or LLAMA_API_KEY:
        try:
            raw_text = await run_in_threadpool(
                llama_api_chat_completion,
                payload.messages,
                model,
                payload.temperature if payload.temperature is not None else 0.1,
                payload.max_tokens or 500,
            )
            payload_result = build_openai_chat_response(raw_text, model)
            return JSONResponse(content=payload_result)
        except RuntimeError:
            pass

    client = build_groq_client(api_key=get_authorization_api_key(request))
    response = client.chat.completions.create(
        model=model,
        messages=openai_messages_to_groq_input(payload.messages),
        max_tokens=payload.max_tokens or 500,
        temperature=payload.temperature if payload.temperature is not None else 0.1,
    )

    raw_text = extract_response_text(response).strip()
    payload_result = build_openai_chat_response(raw_text, model)
    return JSONResponse(content=payload_result)


@app.get("/", tags=["health"])
async def root():
    return {"service": "elephante-ai-server", "status": "ok"}


@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "encoder": encoder_type,
        "match_head": match_head is not None,
        "groq_configured": bool(GROQ_API_KEY),
        "hf_local_inference": HF_LOCAL_INFERENCE,
        "sam": sam_segmenter.status(),
    }


@app.post("/analyze")
async def analyze_image(payload: AnalyzeRequest):
    # 1. Resolve the image from either contract.
    image = None
    try:
        if payload.image_base64:
            image = parse_base64_image(payload.image_base64)
        elif payload.image_url:
            image = await run_in_threadpool(fetch_image_from_url, payload.image_url)
        else:
            return JSONResponse(
                content={"error": "Provide image_base64 or image_url.", "status": "failed"},
                status_code=400,
            )
    except ValueError as exc:
        # Bad base64 is a client error; a fetch failure degrades to
        # metadata-only analysis so the web app never breaks.
        if payload.image_base64:
            return JSONResponse(content={"error": str(exc), "status": "failed"}, status_code=400)
        logger.info("analyze: image fetch rejected (%s); metadata-only analysis.", exc)
    except Exception as exc:
        logger.warning("analyze: image fetch failed (%s); metadata-only analysis.", exc)

    # 2. Analyze with Groq, falling back to heuristics on any failure.
    if image is not None:
        try:
            result = await run_in_threadpool(analyze_with_groq, image)
            return JSONResponse(content=enrich_analysis_for_client(result, payload, "groq"))
        except Exception as exc:
            logger.warning("analyze: Groq failed (%s); using heuristic fallback.", exc)

    return JSONResponse(
        content=enrich_analysis_for_client(heuristic_analysis(payload), payload, "heuristic")
    )
