"""
One-Click Apple CoreML Exporter for Multi-Task Volumetric PoseNet.
Exports PyTorch weights to native Float16 CoreML .mlpackage optimized for Apple Neural Engine (ANE).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model import MultiTaskVolumetricPoseNet


def export_to_coreml(
    checkpoint_path: str,
    output_mlpackage: str = "VolumetricPoseNet_ANE.mlpackage",
    backbone: str = "mobilenet_v3_large"
):
    try:
        import coremltools as ct
    except ImportError:
        print("❌ 'coremltools' not installed. Run on macOS: pip install coremltools")
        return

    print("======================================================================")
    print("🍏 EXPORTING MULTI-TASK VOLUMETRIC POSENET TO APPLE COREML")
    print("======================================================================")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output File : {output_mlpackage}")
    print("======================================================================")

    # 1. Load PyTorch model
    device = torch.device("cpu")
    model = MultiTaskVolumetricPoseNet(backbone_name=backbone, pretrained=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # 2. Trace with TorchScript
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(model, dummy_input)

    # 3. Convert to CoreML MLProgram
    print("Converting to Apple CoreML MLProgram with ANE optimization...")
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 256, 256),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[
            ct.TensorType(name="coordinates_17"),
            ct.TensorType(name="volume_radii_6"),
            ct.TensorType(name="visibility_17")
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL  # Enables Apple Neural Engine (ANE) + GPU
    )

    # 4. Save .mlpackage
    output_path = Path(output_mlpackage)
    mlmodel.save(str(output_path))
    print(f"🎉 SUCCESS! CoreML package saved to: {output_path.resolve()}")
    print("Ready to drop into Xcode project (Sketsain iPadOS)!")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = str(PROJECT_ROOT / "checkpoints_dgx" / "volumetric_posenet_best.pth")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "VolumetricPoseNet_ANE.mlpackage")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Volumetric PoseNet to CoreML")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                        help="Path to trained PyTorch .pth checkpoint")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output .mlpackage filename")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large",
                        help="Backbone architecture matching checkpoint")
    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.output, args.backbone)
