import base64
import colorsys
import io
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
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

app = FastAPI(title="Elephante AI Stylist API")

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
