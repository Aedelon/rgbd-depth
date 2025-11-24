#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Command-line interfaces for Camera Depth Models."""

import argparse
import sys
from pathlib import Path


def main_download():
    """CLI for downloading pre-trained models from HuggingFace."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained Camera Depth Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download RealSense D435 model
  cdm-download --camera d435

  # Download to specific directory
  cdm-download --camera d435 --output-dir ./my_models

  # List available models
  cdm-download --list
        """,
    )

    parser.add_argument(
        "--camera",
        type=str,
        choices=[
            "d405",
            "d415",
            "d435",
            "d455",
            "l515",  # RealSense
            "zed2i-quality",
            "zed2i-neural",  # ZED 2i
            "kinect",  # Azure Kinect
        ],
        help="Camera model to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)",
    )
    parser.add_argument("--list", action="store_true", help="List all available models")

    args = parser.parse_args()

    # Model repository mapping
    MODELS = {
        "d405": "depth-anything/camera-depth-model-d405",
        "d415": "depth-anything/camera-depth-model-d415",
        "d435": "depth-anything/camera-depth-model-d435",
        "d455": "depth-anything/camera-depth-model-d455",
        "l515": "depth-anything/camera-depth-model-l515",
        "zed2i-quality": "depth-anything/camera-depth-model-zed2i-quality",
        "zed2i-neural": "depth-anything/camera-depth-model-zed2i-neural",
        "kinect": "depth-anything/camera-depth-model-kinect",
    }

    if args.list:
        print("Available Camera Depth Models:")
        print("\nIntel RealSense:")
        for cam in ["d405", "d415", "d435", "d455", "l515"]:
            print(f"  - {cam}")
        print("\nStereolabs ZED 2i:")
        for cam in ["zed2i-quality", "zed2i-neural"]:
            print(f"  - {cam}")
        print("\nMicrosoft Azure Kinect:")
        print("  - kinect")
        print("\nUsage: cdm-download --camera <model_name>")
        return

    if not args.camera:
        parser.error("--camera is required (or use --list to see available models)")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        print("Or: pip install camera-depth-models[download]")
        sys.exit(1)

    print(f"Downloading {args.camera} model from HuggingFace...")
    print(f"Repository: {MODELS[args.camera]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_path = hf_hub_download(
            repo_id=MODELS[args.camera],
            filename="model.pth",
            cache_dir=str(output_dir),
            resume_download=True,
        )
        print("✓ Model downloaded successfully!")
        print(f"  Path: {model_path}")
        print("\nUsage:")
        print(
            f"  cdm-infer --encoder vitl --model-path {model_path} --rgb-image <rgb> --depth-image <depth>"
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


def main_infer():
    """CLI for running inference with Camera Depth Models."""
    # Import the original infer script's main logic
    try:
        # We'll create a separate infer.py that this imports from
        from rgbddepth.infer import main as run_inference

        run_inference()
    except ImportError:
        print("Error: Inference module not found")
        print("This should not happen in a proper installation")
        sys.exit(1)


if __name__ == "__main__":
    # For testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        main_download()
    else:
        main_infer()
