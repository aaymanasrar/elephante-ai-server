# SAM Integration

This API can optionally use Meta Segment Anything (SAM) to isolate garments
before color analysis and visual matching. The integration is lazy-loaded, so
the server still starts without a SAM checkpoint.

## Setup

Install dependencies:

```bash
venv/bin/pip install -r requirements.txt
```

Download a SAM checkpoint from Meta's official repository:

https://github.com/facebookresearch/segment-anything#model-checkpoints

For the fastest local setup, use the ViT-B checkpoint. If the file is stored at
`checkpoints/sam_vit_b_01ec64.pth`, the API detects it automatically.

You can also point the API at a custom checkpoint:

```bash
export SAM_CHECKPOINT=/absolute/path/to/sam_vit_b_01ec64.pth
export SAM_MODEL_TYPE=vit_b
export SAM_DEVICE=cpu
```

`SAM_DEVICE` can also be `cuda` or `mps` when the local PyTorch install
supports it.

Start the API as usual:

```bash
venv/bin/uvicorn main:app --reload
```

## Endpoints

`GET /sam-status`

Returns whether SAM is configured, which checkpoint is selected, and whether
the model has been loaded into memory yet.

`POST /segment-garment`

Form fields:

- `image`: required upload.
- `box`: optional `x1,y1,x2,y2` prompt. If omitted, the server estimates a
  foreground garment box from the image border background.
- `point_coords`: optional `x,y;x,y` prompt points.
- `point_labels`: optional comma-separated labels matching `point_coords`;
  `1` is foreground and `0` is background.
- `crop`: optional boolean, defaults to `true`.
- `require_segmentation`: optional boolean, defaults to `false`. When false,
  the endpoint returns the original image as a successful fallback instead of
  failing hard if SAM is not configured or the mask is not usable.

Returns mask metadata plus `mask_png_base64`, `isolated_png_base64`, and a
`segmentation_status` value.

`POST /predict-match`

The existing endpoint now accepts optional `use_segmentation`, defaulting to
`true`. If SAM is configured, shirt and pants images are masked and cropped
before embeddings and color analysis. If SAM is missing or fails, the endpoint
falls back to the original image and reports the reason under `segmentation`.

## ONNX

ONNX is a model interchange format. In Python, `onnxruntime` is the runtime
used to execute exported ONNX graphs. Meta's official SAM ONNX path exports the
prompt encoder and mask decoder, while the heavier ViT image encoder remains in
PyTorch.

Export the local ViT-B SAM decoder:

```bash
venv/bin/python scripts/export_sam_onnx.py
```

This writes `checkpoints/sam_vit_b_mask_decoder.onnx` and validates it with
`onnxruntime`.
