#!/usr/bin/env python3
"""Export pipeline: pth -> safetensors -> ONNX -> TRT FP16 -> TRT FP32.

Usage:
    python scripts/export.py --checkpoint /path/to/best.pth --config configs/paper.toml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from anima_tde.model import build_model
from anima_tde.utils import load_config


def export_safetensors(model: torch.nn.Module, output_path: Path) -> None:
    """Export model to safetensors format."""
    try:
        from safetensors.torch import save_file

        state = dict(model.state_dict().items())
        save_file(state, str(output_path))
        print(f"[EXPORT] safetensors: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    except ImportError:
        print("[WARN] safetensors not installed, skipping")


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    opset: int = 17,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["det_p3", "det_p4", "det_p5"],
        dynamic_axes={
            "image": {0: "batch"},
            "det_p3": {0: "batch"},
            "det_p4": {0: "batch"},
            "det_p5": {0: "batch"},
        },
    )
    print(f"[EXPORT] ONNX: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Validate
    try:
        import onnx

        model_onnx = onnx.load(str(output_path))
        onnx.checker.check_model(model_onnx)
        print("[EXPORT] ONNX validation passed")
    except Exception as e:
        print(f"[WARN] ONNX validation failed: {e}")


def export_tensorrt(
    onnx_path: Path,
    output_path: Path,
    fp16: bool = False,
    workspace_gb: int = 4,
) -> None:
    """Export ONNX to TensorRT engine.

    First tries shared TRT toolkit, then falls back to trtexec.
    """
    # Try shared TRT toolkit
    shared_trt = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    precision = "fp16" if fp16 else "fp32"

    if shared_trt.exists():
        import subprocess

        cmd = [
            sys.executable, str(shared_trt),
            "--onnx", str(onnx_path),
            "--output", str(output_path),
            "--precision", precision,
            "--workspace", str(workspace_gb),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[EXPORT] TensorRT {precision}: {output_path}")
            return
        print(f"[WARN] Shared TRT toolkit failed: {result.stderr[:200]}")

    # Fallback: try trtexec
    import shutil

    trtexec = shutil.which("trtexec")
    if trtexec:
        import subprocess

        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={output_path}",
            f"--workspace={workspace_gb * 1024}",
        ]
        if fp16:
            cmd.append("--fp16")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[EXPORT] TensorRT {precision}: {output_path}")
            return
        print(f"[WARN] trtexec failed: {result.stderr[:200]}")

    # Fallback: try torch_tensorrt
    try:
        import torch_tensorrt

        model_onnx = torch.jit.load(str(onnx_path))
        dtype = torch.float16 if fp16 else torch.float32
        trt_model = torch_tensorrt.compile(
            model_onnx,
            inputs=[torch_tensorrt.Input(shape=[1, 3, 640, 640], dtype=dtype)],
            enabled_precisions={dtype},
        )
        torch.jit.save(trt_model, str(output_path))
        print(f"[EXPORT] TensorRT (torch_tensorrt) {precision}: {output_path}")
    except Exception as e:
        print(f"[SKIP] TensorRT {precision} export not available: {e}")
        # Create placeholder
        output_path.write_text(
            f"# TensorRT {precision} export placeholder\n"
            f"# Build manually: trtexec --onnx={onnx_path} --saveEngine={output_path}"
            + (" --fp16" if fp16 else "") + "\n"
        )
        print(f"[SKIP] Created placeholder at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TDE Export Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/paper.toml")
    parser.add_argument(
        "--output-dir", type=str,
        default="/mnt/artifacts-datai/exports/project_tde",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[LOAD] Model from {args.checkpoint}")
    print(f"[LOAD] Parameters: {model.count_parameters() / 1e6:.2f}M")

    # 1. PyTorch checkpoint (already have it, just copy)
    pth_path = output_dir / "tde_best.pth"
    torch.save(ckpt, pth_path)
    print(f"[EXPORT] PyTorch: {pth_path}")

    # 2. Safetensors
    safe_path = output_dir / "tde_best.safetensors"
    export_safetensors(model, safe_path)

    # 3. ONNX
    in_ch = config["model"].get("input_channels", 3)
    h = config["model"].get("input_height", 640)
    w = config["model"].get("input_width", 640)
    onnx_path = output_dir / "tde_best.onnx"
    export_onnx(model, onnx_path, input_shape=(1, in_ch, h, w))

    # 4. TensorRT FP16
    trt_fp16_path = output_dir / "tde_best_fp16.engine"
    export_tensorrt(onnx_path, trt_fp16_path, fp16=True)

    # 5. TensorRT FP32
    trt_fp32_path = output_dir / "tde_best_fp32.engine"
    export_tensorrt(onnx_path, trt_fp32_path, fp16=False)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / 1e6
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
