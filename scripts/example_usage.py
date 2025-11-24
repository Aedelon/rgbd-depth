#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of the optimized CDM model.
Demonstrates both automatic and manual configuration.
"""

import torch
import numpy as np
from rgbddepth.dpt import RGBDDepth
from rgbddepth.optimization_config import (
    OptimizationConfig,
    get_optimal_config,
    get_config_for_inference
)


def example_auto_config():
    """Example 1: Automatic configuration (recommended for most users)."""
    print("\n" + "="*60)
    print("Example 1: Automatic Configuration")
    print("="*60)

    # Simply create a config with device='auto'
    # Everything else is automatically configured
    config = OptimizationConfig(device="auto")

    print(config.summary())

    # Create model
    model = RGBDDepth(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        config=config
    )

    print("\n✓ Model created with automatic optimizations")


def example_manual_config():
    """Example 2: Manual configuration for fine-grained control."""
    print("\n" + "="*60)
    print("Example 2: Manual Configuration")
    print("="*60)

    # Detect available device for this example
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        backend = "xformers"  # or "torch" if xformers not available
        precision = "fp16"
        use_compile = True
    elif torch.backends.mps.is_available():
        device = "mps"
        backend = "manual"  # Better than torch on MPS
        precision = "fp32"  # FP32 is faster on MPS for this model
        use_compile = False
    else:
        device = "cpu"
        backend = "torch"
        precision = "fp32"
        use_compile = False

    # Manually specify all optimization parameters
    config = OptimizationConfig(
        device=device,
        attention_backend=backend,
        use_compile=use_compile,
        use_channels_last=True,  # Use channels_last memory format
        mixed_precision=precision,
        fuse_depth_encoder=False,  # Keep separate encoders
        interpolation_mode="bilinear"  # Good balance
    )

    print(config.summary())

    # Create model
    model = RGBDDepth(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        config=config
    )

    print("\n✓ Model created with manual configuration")


def example_device_specific():
    """Example 3: Device-specific optimal configurations."""
    print("\n" + "="*60)
    print("Example 3: Device-Specific Configurations")
    print("="*60)

    # Get optimal config for each device type
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")

    for device in devices:
        print(f"\n--- Optimal config for {device.upper()} ---")
        config = get_optimal_config(device=device)
        print(config.summary())


def example_preset_profiles():
    """Example 4: Using preset profiles."""
    print("\n" + "="*60)
    print("Example 4: Preset Profiles")
    print("="*60)

    # Profile 1: Speed-optimized inference
    print("\n--- Speed-Optimized Inference ---")
    config = get_config_for_inference(device="auto", prefer_quality=False)
    print(config.summary())

    # Profile 2: Quality-optimized inference
    print("\n--- Quality-Optimized Inference ---")
    config = get_config_for_inference(device="auto", prefer_quality=True)
    print(config.summary())


def example_compare_backends():
    """Example 5: Comparing attention backends."""
    print("\n" + "="*60)
    print("Example 5: Comparing Attention Backends")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backends = ["torch", "manual"]
    if device == "cuda":
        try:
            import xformers
            backends.insert(0, "xformers")
        except ImportError:
            print("Note: xformers not available, skipping xformers comparison")

    for backend in backends:
        print(f"\n--- Backend: {backend} ---")
        try:
            config = OptimizationConfig(
                device=device,
                attention_backend=backend
            )

            model = RGBDDepth(
                encoder="vits",  # Use smaller model for quick testing
                features=64,
                out_channels=[48, 96, 192, 384],
                config=config
            )
            print(f"✓ {backend} backend initialized successfully")
        except Exception as e:
            print(f"✗ {backend} backend failed: {e}")


def example_full_inference():
    """Example 6: Full inference pipeline with optimizations."""
    print("\n" + "="*60)
    print("Example 6: Full Inference Pipeline")
    print("="*60)

    # Create optimized model
    config = get_config_for_inference(device="auto")
    print(config.summary())

    model = RGBDDepth(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        config=config
    )
    model.eval()

    # Create dummy data
    print("\n1. Creating dummy RGB and depth images...")
    rgb = np.random.rand(480, 640, 3).astype(np.float32) * 255
    depth = np.random.rand(480, 640).astype(np.float32)
    depth = np.where(depth > 0.1, 1.0 / (depth + 0.1), 0.0)  # Inverse depth

    # Run inference
    print("2. Running inference...")
    with torch.no_grad():
        pred_depth = model.infer_image(rgb, depth, input_size=518)

    print(f"3. Prediction complete!")
    print(f"   Input shape: {rgb.shape}")
    print(f"   Output shape: {pred_depth.shape}")
    print(f"   Depth range: [{pred_depth.min():.3f}, {pred_depth.max():.3f}]")

    # Convert from inverse depth to depth
    pred_depth_meters = 1.0 / (pred_depth + 1e-6)
    print(f"   Depth (meters): [{pred_depth_meters.min():.3f}, {pred_depth_meters.max():.3f}]")

    print("\n✓ Full inference pipeline completed successfully")


def example_memory_efficient():
    """Example 7: Memory-efficient configuration for large models."""
    print("\n" + "="*60)
    print("Example 7: Memory-Efficient Configuration")
    print("="*60)

    # Configuration for when you're running out of memory
    config = OptimizationConfig(
        device="auto",
        mixed_precision="fp16",  # Use FP16 to save memory
        fuse_depth_encoder=True,  # Share encoder weights
        use_channels_last=True,  # Better memory layout
    )

    print(config.summary())

    print("\n💡 Tips for reducing memory usage:")
    print("  1. Use mixed_precision='fp16' (2x memory savings on GPU)")
    print("  2. Set fuse_depth_encoder=True (shares encoder weights)")
    print("  3. Reduce input_size (518 → 448 → 392)")
    print("  4. Use smaller encoder (vitl → vitb → vits)")
    print("  5. Enable gradient checkpointing for training")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# CDM Optimization Examples")
    print("#"*60)

    examples = [
        ("Automatic Configuration", example_auto_config),
        ("Manual Configuration", example_manual_config),
        ("Device-Specific Configs", example_device_specific),
        ("Preset Profiles", example_preset_profiles),
        ("Compare Backends", example_compare_backends),
        ("Full Inference Pipeline", example_full_inference),
        ("Memory-Efficient Config", example_memory_efficient),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "#"*60)
    print("# All examples completed!")
    print("#"*60)


if __name__ == "__main__":
    main()
