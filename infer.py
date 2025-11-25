#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rgbddepth.dpt import RGBDDepth

# Device will be set after parsing arguments
DEVICE = None

# Model configurations for different Vision Transformer (ViT) encoder sizes
# Each config specifies the encoder type, feature dimensions, and output channels
model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },  # Small
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },  # Base
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },  # Large
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },  # Giant
}


def colorize(value, vmin=None, vmax=None, cmap="Spectral"):
    """Convert depth values to colorized visualization.

    Args:
        value: Input depth array
        vmin: Minimum value for normalization (auto-computed if None)
        vmax: Maximum value for normalization (auto-computed if None)
        cmap: Colormap name for visualization

    Returns:
        RGB image array for visualization
    """
    # Skip processing if input is already RGB
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]  # Extract single channel

    # Mark invalid/zero depth values
    invalid_mask = value < 0.0001

    # Normalize values to [0, 1] range
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = ((value - vmin) / (vmax - vmin)).clip(0, 1)

    # Apply colormap to create colored visualization
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # Returns RGBA values

    # Set invalid regions to black
    value[invalid_mask] = 0

    # Return only RGB channels (drop alpha)
    img = value[..., :3]
    return img


def image_grid(imgs, rows, cols):
    """Create a grid layout from a list of images.

    Args:
        imgs: List of image arrays to arrange in grid
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        Combined image array with grid layout
    """
    if not len(imgs):
        return None

    # Ensure we have the right number of images for the grid
    assert len(imgs) == rows * cols

    # Use dimensions from first image as reference
    h, w = imgs[0].shape[:2]

    # Create empty canvas for the grid
    grid = Image.new("RGB", size=(cols * w, rows * h))

    # Place each image in its grid position
    for i, img in enumerate(imgs):
        # Calculate grid position (row, col) from linear index
        col_idx = i % cols
        row_idx = i // cols

        # Paste image at calculated position
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize((w, h), resample=Image.BILINEAR),
            box=(col_idx * w, row_idx * h),
        )

    return np.array(grid)


def parse_arguments():
    """Parse command line arguments for RGBD depth inference.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="RGBD Depth Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument("--rgb-image", type=str, required=True, help="Path to the RGB input image")
    parser.add_argument(
        "--depth-image", type=str, required=True, help="Path to the depth input image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization image path",
    )
    parser.add_argument("--input-size", type=int, default=518, help="Input size for inference")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument("--max-depth", type=float, default=6.0, help="Maximum valid depth value")
    parser.add_argument("--image-min", type=float, default=0.1, help="Minimum valid depth value")
    parser.add_argument("--image-max", type=float, default=5.0, help="Maximum valid depth value")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for inference (auto=automatic selection)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimizations on CUDA (optimizations are auto-enabled on CUDA only)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision: fp32 (default), fp16 (CUDA/MPS), bf16 (CUDA only, better stability)",
    )
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments and create output directory if needed.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If any required input files don't exist
    """
    # Check if model checkpoint file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)

    # Check if input RGB image exists
    if not os.path.exists(args.rgb_image):
        print(f"Error: RGB image '{args.rgb_image}' does not exist")
        sys.exit(1)

    # Check if input depth image exists
    if not os.path.exists(args.depth_image):
        print(f"Error: Depth image '{args.depth_image}' does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def apply_optimizations(model, device, optimize=False):
    """Apply device-specific optimizations while keeping FP32 precision.

    Args:
        model: The model to optimize
        device: Target device ('cuda', 'mps', 'cpu')
        optimize: Whether to apply optimizations

    Returns:
        Optimized model
    """
    if not optimize:
        return model

    print("\nApplying optimizations (FP32 precision maintained):")

    # torch.compile (PyTorch 2.0+, works on CUDA and CPU, experimental on MPS)
    if torch.__version__ >= "2.0":
        if device == "cuda":
            # Best on CUDA
            try:
                model = torch.compile(model, mode="max-autotune", fullgraph=False)
                print("  ✓ torch.compile enabled (mode=max-autotune)")
            except Exception as e:
                print(f"  ✗ torch.compile failed: {e}")
        elif device == "cpu":
            # CPU benefits from compile but with different mode
            try:
                model = torch.compile(model, mode="default", fullgraph=False)
                print("  ✓ torch.compile enabled (mode=default)")
            except Exception as e:
                print(f"  ✗ torch.compile failed: {e}")
        elif device == "mps":
            # MPS support is experimental for torch.compile
            print("  ⚠ torch.compile not enabled on MPS (experimental support)")
    else:
        print(f"  ⚠ torch.compile requires PyTorch 2.0+ (current: {torch.__version__})")

    # SDPA (Scaled Dot Product Attention) - native in PyTorch 2.0+
    # Already used by FlexibleCrossAttention/nn.MultiheadAttention in PyTorch 2.0+
    if torch.__version__ >= "2.0":
        print("  ✓ SDPA (native optimized attention) available")

    print()
    return model


def load_model(encoder, model_path, optimize=False):
    """Load and initialize the RGBD depth estimation model.

    Args:
        encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
        model_path: Path to the model checkpoint file
        optimize: Whether to apply optimizations

    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    # Detect if xFormers is available (CUDA only)
    use_xformers = False
    if optimize and DEVICE == "cuda":
        try:
            import xformers

            use_xformers = True
            print("✓ xFormers detected, will be used for cross-attention")
        except ImportError:
            print("⚠ xFormers not installed, using SDPA (still fast)")

    # Initialize model with configuration for specified encoder
    config = model_configs[encoder].copy()
    config["use_xformers"] = use_xformers
    model = RGBDDepth(**config)

    # Load checkpoint and extract state dict
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model" in checkpoint:
        # Handle checkpoints that wrap state dict in 'model' key
        # Remove 'module.' prefix if present (from DataParallel training)
        states = {k[7:]: v for k, v in checkpoint["model"].items()}
    elif "state_dict" in checkpoint:
        states = checkpoint["state_dict"]
        states = {k[9:]: v for k, v in states.items()}
    else:
        # Direct state dict checkpoint
        states = checkpoint

    # Load weights and move to device
    model.load_state_dict(states, strict=False)
    model = model.to(DEVICE).eval()

    # Apply optimizations if requested
    model = apply_optimizations(model, DEVICE, optimize)

    print(f"Model loaded: {encoder} from {model_path}")
    return model


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    """Load and preprocess RGB and depth images.

    Args:
        rgb_path: Path to RGB image file
        depth_path: Path to depth image file
        depth_scale: Scale factor to convert depth values to meters
        max_depth: Maximum valid depth value (values above this are set to 0)

    Returns:
        tuple: (rgb_image, depth_low_res, similarity_depth_low_res)
            - rgb_image: RGB image in numpy array format (BGR -> RGB)
            - depth_low_res: Depth values in meters
            - similarity_depth_low_res: Inverse depth values (1/depth) for model input
    """
    # Load RGB image and convert from BGR to RGB
    rgb_src = np.asarray(cv2.imread(rgb_path)[:, :, ::-1])
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")

    # Load depth image (usually 16-bit)
    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    # Convert depth to meters and clamp invalid values
    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    depth_low_res[depth_low_res > max_depth] = 0.0  # Remove values beyond max range

    # Create similarity depth (inverse depth) for model input
    # Only compute inverse for valid depth values
    simi_depth_low_res = np.zeros_like(depth_low_res)
    simi_depth_low_res[depth_low_res > 0] = 1 / depth_low_res[depth_low_res > 0]

    print(f"Images loaded: RGB {rgb_src.shape}, Depth {depth_low_res.shape}")
    return rgb_src, depth_low_res, simi_depth_low_res


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm_func = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm_func(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np


def create_visualization(rgb_src, depth_rs, simi_depth, pred, image_min, image_max):
    """Create a 2x2 grid visualization comparing input and predicted depth.

    Args:
        rgb_src: Original RGB image
        depth_rs: Ground truth depth image
        simi_depth: Similarity (inverse) depth input
        pred: Predicted depth from model

    Returns:
        numpy.ndarray: Combined visualization image with 2x2 grid layout:
            - Top-left: Original RGB image
            - Top-right: Input depth (colorized)
            - Bottom-left: Predicted depth (colorized)
            - Bottom-right: Relative error map (colorized)
    """
    # Convert RGB image to BGR format for display
    rgb_display = cv2.cvtColor(rgb_src, cv2.COLOR_RGB2BGR)

    # Use colorize_depth_maps function to colorize depth maps
    # Colorize predicted depth map
    pred_colored = colorize_depth_maps(
        pred, min_depth=image_min, max_depth=image_max, cmap="Spectral"
    )
    pred_colored = np.rollaxis(pred_colored[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    pred_colored = (pred_colored * 255).astype(np.uint8)

    # Colorize ground truth depth map
    depth_colored = colorize_depth_maps(
        depth_rs, min_depth=image_min, max_depth=image_max, cmap="Spectral"
    )
    depth_colored = np.rollaxis(depth_colored[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Calculate relative error
    depth_error = np.zeros_like(depth_rs)
    valid = depth_rs > 0  # Only calculate error for valid depth pixels
    depth_error[valid] = np.abs(depth_rs[valid] - pred[valid]) / depth_rs[valid]

    # Colorize error map
    error_colored = colorize_depth_maps(depth_error, min_depth=0, max_depth=1, cmap="Spectral")
    error_colored = np.rollaxis(error_colored[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    error_colored = (error_colored * 255).astype(np.uint8)

    # Arrange all visualizations in a 2x2 grid
    return image_grid([rgb_display, depth_colored, pred_colored, error_colored], 2, 2)


def inference(args):
    """Run complete depth inference pipeline.

    Args:
        args: Parsed command line arguments containing all parameters
    """
    # Validate all input files and create output directory
    validate_inputs(args)

    # Validate precision choice
    if args.precision == "bf16" and DEVICE != "cuda":
        print("Warning: BF16 only supported on CUDA, falling back to FP32")
        args.precision = "fp32"
    if args.precision == "fp16" and DEVICE == "cpu":
        print("Warning: FP16 not recommended on CPU, falling back to FP32")
        args.precision = "fp32"

    # Load the trained model
    model = load_model(args.encoder, args.model_path, args.optimize)

    # Load and preprocess input images
    rgb_src, depth_low_res, simi_depth_low_res = load_images(
        args.rgb_image, args.depth_image, args.depth_scale, 25.0
    )

    # Determine autocast dtype
    if args.precision == "fp16":
        dtype = torch.float16
        print("Using mixed precision: FP16")
    elif args.precision == "bf16":
        dtype = torch.bfloat16
        print("Using mixed precision: BF16")
    else:
        dtype = None  # No autocast, full FP32
        print("Using full precision: FP32")

    # Run model inference with optional mixed precision
    if dtype is not None:
        # Use autocast for mixed precision
        device_type = "cuda" if DEVICE == "cuda" else "cpu"  # MPS uses cpu context
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            pred = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
    else:
        # Full precision
        pred = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)

    print(f"Prediction info: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}")

    # Convert from inverse depth back to regular depth
    pred = 1 / pred

    # Create visualization comparing input, prediction, and error
    image_min = args.image_min
    image_max = args.image_max
    artifact = create_visualization(
        rgb_src, depth_low_res, simi_depth_low_res, pred, image_min, image_max
    )

    # Save the visualization
    Image.fromarray(artifact).save(args.output)
    print(f"Output saved to: {args.output}")


def main():
    """Main entry point for the rgbd-depth CLI."""
    # Parse command line arguments
    args = parse_arguments()

    # Set device based on user choice
    global DEVICE
    if args.device == "auto":
        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        DEVICE = args.device
        # Validate device availability
        if DEVICE == "cuda" and not torch.cuda.is_available():
            print("Error: CUDA requested but not available")
            sys.exit(1)
        if DEVICE == "mps" and not torch.backends.mps.is_available():
            print("Error: MPS requested but not available")
            sys.exit(1)

    print(f"Using device: {DEVICE}")

    # Auto-enable optimizations on CUDA (unless --no-optimize is set)
    args.optimize = (DEVICE == "cuda") and not args.no_optimize

    # Run inference
    inference(args)


if __name__ == "__main__":
    main()
