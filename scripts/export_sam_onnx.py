import argparse
import warnings
from pathlib import Path

import torch
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel


DEFAULT_CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
DEFAULT_OUTPUT = Path("checkpoints/sam_vit_b_mask_decoder.onnx")


def model_type_from_checkpoint(path: Path) -> str:
    name = path.name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    return "vit_b"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export SAM's prompt encoder and mask decoder to ONNX. "
            "The ViT image encoder still runs in PyTorch in Meta's official ONNX flow."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--multi-mask", action="store_true")
    parser.add_argument("--gelu-approximate", action="store_true")
    parser.add_argument("--use-stability-score", action="store_true")
    return parser.parse_args()


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def build_dummy_inputs(sam):
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * value for value in embed_size]
    return {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }


def export_sam_onnx(
    checkpoint: Path,
    output: Path,
    model_type: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool,
    use_stability_score: bool,
):
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM checkpoint: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
    )

    if gelu_approximate:
        for module in onnx_model.modules():
            if isinstance(module, torch.nn.GELU):
                module.approximate = "tanh"

    dummy_inputs = build_dummy_inputs(sam)
    _ = onnx_model(**dummy_inputs)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    print(f"Exporting ONNX model: {output}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            str(output),
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print("Validating ONNX graph...")
    import onnx

    onnx_model_proto = onnx.load(str(output))
    onnx.checker.check_model(onnx_model_proto)

    print("Running ONNX Runtime smoke test...")
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(output),
        providers=["CPUExecutionProvider"],
    )
    session.run(None, {key: to_numpy(value) for key, value in dummy_inputs.items()})

    print("Done.")


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    model_type = args.model_type or model_type_from_checkpoint(checkpoint)
    export_sam_onnx(
        checkpoint=checkpoint,
        output=args.output.expanduser(),
        model_type=model_type,
        opset=args.opset,
        return_single_mask=not args.multi_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
    )


if __name__ == "__main__":
    main()
