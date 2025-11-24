#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Inference script with device-specific optimizations."""

import argparse
import os
import sys
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rgbddepth.dpt import RGBDDepth
from rgbddepth.optimization_config import OptimizationConfig

# Model configurations
model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RGBD Depth Inference with Device-Specific Optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
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

    # Input/output arguments
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

    # Depth processing arguments
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument("--max-depth", type=float, default=6.0, help="Maximum valid depth value")
    parser.add_argument(
        "--image-min", type=float, default=0.1, help="Minimum depth for visualization"
    )
    parser.add_argument(
        "--image-max", type=float, default=5.0, help="Maximum depth for visualization"
    )

    # Optimization arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=["auto", "xformers", "torch", "manual"],
        help="Attention backend to use",
    )
    parser.add_argument(
        "--use-compile",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Enable torch.compile (GPU only)",
    )
    parser.add_argument(
        "--use-channels-last",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Use channels_last memory format",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--fuse-encoder",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Fuse RGB and depth encoders",
    )
    parser.add_argument(
        "--interpolation-mode",
        type=str,
        default="bilinear",
        choices=["bicubic", "bilinear", "nearest"],
        help="Interpolation mode for decoder upsampling",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
        help="Number of iterations for benchmarking",
    )

    return parser.parse_args()


def colorize(value, vmin=None, vmax=None, cmap="Spectral"):
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]

    invalid_mask = value < 0.0001

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = ((value - vmin) / (vmax - vmin)).clip(0, 1)

    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    value[invalid_mask] = 0
    img = value[..., :3]
    return img


def image_grid(imgs, rows, cols):
    if not len(imgs):
        return None
    assert len(imgs) == rows * cols
    h, w = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        col_idx = i % cols
        row_idx = i // cols
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize((w, h), resample=Image.BILINEAR),
            box=(col_idx * w, row_idx * h),
        )
    return np.array(grid)


def load_model(args):
    def parse_opt_bool(val):
        if val in (None, "auto"):
            return None
        if isinstance(val, str):
            return val.lower() in ["true", "1", "yes", "y"]
        return bool(val)

    config = OptimizationConfig(
        device=args.device,
        attention_backend=args.attention_backend,
        use_compile=parse_opt_bool(args.use_compile),
        use_channels_last=parse_opt_bool(args.use_channels_last),
        mixed_precision=args.mixed_precision,
        fuse_depth_encoder=parse_opt_bool(args.fuse_encoder),
        interpolation_mode=args.interpolation_mode,
    )

    model = RGBDDepth(**model_configs[args.encoder], config=config)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if "model" in checkpoint:
        states = {k[7:]: v for k, v in checkpoint["model"].items()}
    elif "state_dict" in checkpoint:
        states = checkpoint["state_dict"]
        states = {k[9:]: v for k, v in states.items()}
    else:
        states = checkpoint

    model.load_state_dict(states, strict=False)
    model.eval()
    print(f"✓ Model loaded: {args.encoder} from {args.model_path}")
    return model


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    rgb_src = np.asarray(cv2.imread(rgb_path)[:, :, ::-1])
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")

    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    depth_low_res[depth_low_res > max_depth] = 0.0

    simi_depth_low_res = np.zeros_like(depth_low_res)
    simi_depth_low_res[depth_low_res > 0] = 1 / depth_low_res[depth_low_res > 0]

    print(f"Images loaded: RGB {rgb_src.shape}, Depth {depth_low_res.shape}")
    return rgb_src, depth_low_res, simi_depth_low_res


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    assert len(depth_map.shape) >= 2, "Invalid dimension"
    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    cm_func = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm_func(depth, bytes=False)[:, :, :, 0:3]
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np


def create_visualization(rgb_src, depth_rs, pred, image_min, image_max):
    rgb_display = cv2.cvtColor(rgb_src, cv2.COLOR_RGB2BGR)

    pred_colored = colorize_depth_maps(
        pred, min_depth=image_min, max_depth=image_max, cmap="Spectral"
    )
    pred_colored = np.rollaxis(pred_colored[0], 0, 3)
    pred_colored = (pred_colored * 255).astype(np.uint8)

    depth_colored = colorize_depth_maps(
        depth_rs, min_depth=image_min, max_depth=image_max, cmap="Spectral"
    )
    depth_colored = np.rollaxis(depth_colored[0], 0, 3)
    depth_colored = (depth_colored * 255).astype(np.uint8)

    depth_error = np.zeros_like(depth_rs)
    valid = depth_rs > 0
    depth_error[valid] = np.abs(depth_rs[valid] - pred[valid]) / depth_rs[valid]

    error_colored = colorize_depth_maps(depth_error, min_depth=0, max_depth=1, cmap="Spectral")
    error_colored = np.rollaxis(error_colored[0], 0, 3)
    error_colored = (error_colored * 255).astype(np.uint8)

    return image_grid([rgb_display, depth_colored, pred_colored, error_colored], 2, 2)


def inference(args):
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)
    if not os.path.exists(args.rgb_image):
        print(f"Error: RGB image '{args.rgb_image}' does not exist")
        sys.exit(1)
    if not os.path.exists(args.depth_image):
        print(f"Error: Depth image '{args.depth_image}' does not exist")
        sys.exit(1)

    model = load_model(args)

    rgb_src, depth_low_res, simi_depth_low_res = load_images(
        args.rgb_image, args.depth_image, args.depth_scale, 25.0
    )

    if args.benchmark:
        warmup = 5
        iters = args.benchmark_iters
        print(f"Benchmarking {iters} iters (warmup {warmup})...")
        for _ in range(warmup):
            _ = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(iters):
            _ = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
        duration = (time.time() - start) / iters * 1000
        print(f"Avg latency: {duration:.2f} ms")

    pred = model.infer_image(rgb_src, simi_depth_low_res, input_size=args.input_size)
    print(f"Prediction info: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}")

    pred = 1 / pred

    image_min = args.image_min
    image_max = args.image_max
    artifact = create_visualization(rgb_src, depth_low_res, pred, image_min, image_max)

    Image.fromarray(artifact).save(args.output)
    print(f"✓ Output saved to: {args.output}")


def main():
    """Main entry point for CLI."""
    args = parse_arguments()
    inference(args)


if __name__ == "__main__":
    main()
