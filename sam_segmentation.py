import importlib.util
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


Box = Tuple[int, int, int, int]
LOCAL_CHECKPOINT_CANDIDATES = (
    Path(__file__).with_name("sam_vit_b_01ec64.pth"),
    Path(__file__).with_name("checkpoints") / "sam_vit_b_01ec64.pth",
    Path(__file__).with_name("models") / "sam_vit_b_01ec64.pth",
)


class SamUnavailableError(RuntimeError):
    """Raised when SAM cannot be used in the current runtime."""


@dataclass
class SamMaskResult:
    mask: np.ndarray
    score: float
    bbox_xyxy: Box
    prompt_box_xyxy: Optional[Box]
    model_type: str
    checkpoint_path: str

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox_xywh(self) -> Tuple[int, int, int, int]:
        left, top, right, bottom = self.bbox_xyxy
        return left, top, max(0, right - left), max(0, bottom - top)

    def to_metadata(self) -> dict:
        return {
            "score": round(float(self.score), 4),
            "area": self.area,
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_xywh": list(self.bbox_xywh),
            "prompt_box_xyxy": (
                list(self.prompt_box_xyxy) if self.prompt_box_xyxy is not None else None
            ),
            "model_type": self.model_type,
            "checkpoint": Path(self.checkpoint_path).name,
        }


def default_sam_device(fallback: str = "cpu") -> str:
    try:
        import torch
    except ImportError:
        return fallback

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return fallback


def default_checkpoint_path() -> Optional[Path]:
    for candidate in LOCAL_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def model_type_from_checkpoint(path: Optional[Path], fallback: str = "vit_b") -> str:
    if path is None:
        return fallback

    name = path.name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"

    return fallback


def clamp_box(box: Sequence[float], width: int, height: int) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain 4 comma-separated values: x1,y1,x2,y2.")

    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(1, min(width, int(round(x2))))
    bottom = max(1, min(height, int(round(y2))))

    if right <= left or bottom <= top:
        raise ValueError("box must have x2 > x1 and y2 > y1.")

    return left, top, right, bottom


def mask_to_box(mask: np.ndarray, padding_ratio: float = 0.0) -> Box:
    height, width = mask.shape[:2]
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, width, height

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    if padding_ratio > 0:
        pad = int(round(max(width, height) * padding_ratio))
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(width, right + pad)
        bottom = min(height, bottom + pad)

    return left, top, right, bottom


def mask_area_ratio(mask: np.ndarray) -> float:
    height, width = mask.shape[:2]
    area = height * width
    if area == 0:
        return 0.0

    return float(mask.sum() / area)


def mask_quality(mask: np.ndarray) -> dict:
    area_ratio = mask_area_ratio(mask)
    usable = 0.002 <= area_ratio <= 0.98

    if area_ratio < 0.002:
        reason = "SAM found only a tiny mask, so the original image was used."
    elif area_ratio > 0.98:
        reason = "SAM selected almost the whole image, so the original image was used."
    else:
        reason = None

    return {
        "usable": usable,
        "area_ratio": round(area_ratio, 4),
        "reason": reason,
    }


def estimate_foreground_box(image: Image.Image, padding_ratio: float = 0.03) -> Box:
    """Estimate a prompt box from pixels that differ from the border background."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb).astype(np.int16)
    height, width = array.shape[:2]

    if height == 0 or width == 0:
        return 0, 0, width, height

    border = np.concatenate(
        [
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.mean(np.abs(array - background), axis=2)
    foreground = distance > 24
    foreground_area = int(foreground.sum())
    image_area = height * width

    if foreground_area < max(16, int(image_area * 0.002)):
        return 0, 0, width, height

    if foreground_area > int(image_area * 0.95):
        return 0, 0, width, height

    return mask_to_box(foreground, padding_ratio=padding_ratio)


def apply_mask(
    image: Image.Image,
    mask: np.ndarray,
    background: Tuple[int, int, int] = (255, 255, 255),
    crop: bool = True,
    padding_ratio: float = 0.04,
) -> Image.Image:
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb).astype(np.uint8)
    background_pixels = np.full_like(pixels, background, dtype=np.uint8)
    isolated = np.where(mask[..., None], pixels, background_pixels)
    isolated_image = Image.fromarray(isolated)

    if crop:
        isolated_image = isolated_image.crop(mask_to_box(mask, padding_ratio=padding_ratio))

    return isolated_image


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class SamSegmenter:
    def __init__(
        self,
        checkpoint_path: Optional[Path],
        model_type: str = "vit_b",
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self._predictor = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, device: Optional[object] = None) -> "SamSegmenter":
        checkpoint = os.getenv("SAM_CHECKPOINT") or os.getenv("SAM_CHECKPOINT_PATH")
        checkpoint_path = Path(checkpoint).expanduser() if checkpoint else default_checkpoint_path()
        model_type = os.getenv("SAM_MODEL_TYPE") or model_type_from_checkpoint(checkpoint_path)
        sam_device = os.getenv("SAM_DEVICE") or str(device or "cpu")
        return cls(checkpoint_path=checkpoint_path, model_type=model_type, device=sam_device)

    def status(self) -> dict:
        if self.checkpoint_path is None:
            return {
                "configured": False,
                "available": False,
                "reason": "Set SAM_CHECKPOINT to a local SAM .pth checkpoint.",
            }

        if not self.checkpoint_path.exists():
            return {
                "configured": True,
                "available": False,
                "reason": f"SAM checkpoint not found: {self.checkpoint_path}",
            }

        if importlib.util.find_spec("segment_anything") is None:
            return {
                "configured": True,
                "available": False,
                "reason": "segment-anything is not installed.",
            }

        return {
            "configured": True,
            "available": True,
            "model_type": self.model_type,
            "device": self.device,
            "checkpoint": self.checkpoint_path.name,
            "loaded": self._predictor is not None,
        }

    def _ensure_loaded(self):
        if self._predictor is not None:
            return

        if self.checkpoint_path is None:
            raise SamUnavailableError(
                "SAM is not configured. Set SAM_CHECKPOINT to a local .pth checkpoint."
            )

        if not self.checkpoint_path.exists():
            raise SamUnavailableError(f"SAM checkpoint not found: {self.checkpoint_path}")

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise SamUnavailableError(
                "segment-anything is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        try:
            sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        except KeyError as exc:
            valid_types = ", ".join(sorted(sam_model_registry.keys()))
            raise SamUnavailableError(
                f"Unknown SAM_MODEL_TYPE '{self.model_type}'. Expected one of: {valid_types}."
            ) from exc

        sam.to(device=self.device)
        sam.eval()
        self._predictor = SamPredictor(sam)

    def segment(
        self,
        image: Image.Image,
        box: Optional[Sequence[float]] = None,
        point_coords: Optional[Iterable[Sequence[float]]] = None,
        point_labels: Optional[Sequence[int]] = None,
    ) -> SamMaskResult:
        rgb = image.convert("RGB")
        image_array = np.asarray(rgb)
        height, width = image_array.shape[:2]
        if box is not None:
            prompt_box = clamp_box(box, width, height)
        else:
            prompt_box = estimate_foreground_box(rgb)

        coords_array = None
        labels_array = None
        if point_coords is not None:
            coords = [[float(x), float(y)] for x, y in point_coords]
            coords_array = np.asarray(coords, dtype=np.float32)
            labels = point_labels if point_labels is not None else [1] * len(coords)
            labels_array = np.asarray(labels, dtype=np.int64)

            if len(coords_array) != len(labels_array):
                raise ValueError("point_coords and point_labels must have the same length.")

        with self._lock:
            self._ensure_loaded()
            self._predictor.set_image(image_array)
            masks, scores, _ = self._predictor.predict(
                point_coords=coords_array,
                point_labels=labels_array,
                box=np.asarray(prompt_box, dtype=np.float32),
                multimask_output=True,
            )

        best_index = int(np.argmax(scores))
        mask = masks[best_index].astype(bool)

        return SamMaskResult(
            mask=mask,
            score=float(scores[best_index]),
            bbox_xyxy=mask_to_box(mask),
            prompt_box_xyxy=prompt_box,
            model_type=self.model_type,
            checkpoint_path=str(self.checkpoint_path),
        )
