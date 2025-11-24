#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Test script to validate optimizations across different configurations.
"""

import time

import numpy as np
import torch

from rgbddepth.dpt import RGBDDepth
from rgbddepth.optimization_config import OptimizationConfig


def create_dummy_inputs(batch_size=1, height=518, width=518):
    """Create dummy RGB and depth inputs for testing."""
    rgb = np.random.rand(height, width, 3).astype(np.float32)
    depth = np.random.rand(height, width).astype(np.float32)

    # Simulate inverse depth
    depth = np.where(depth > 0.1, 1.0 / (depth + 0.1), 0.0)

    return rgb, depth


def test_configuration(encoder="vitl", config=None, num_warmup=3, num_iters=10):
    """Test a specific configuration."""
    print(f"\n{'='*60}")
    print("Testing Configuration:")
    print(config.summary())

    try:
        # Create model
        model = RGBDDepth(
            encoder=encoder, features=256, out_channels=[256, 512, 1024, 1024], config=config
        )
        model.eval()

        # Create dummy inputs
        rgb, depth = create_dummy_inputs()

        # Warmup
        print(f"\nWarming up ({num_warmup} iterations)...")
        for i in range(num_warmup):
            try:
                with torch.no_grad():
                    _ = model.infer_image(rgb, depth, input_size=518)
            except Exception as e:
                print(f"  ⚠ Warmup iteration {i+1} failed: {e}")
                return None

        # Benchmark
        print(f"Running benchmark ({num_iters} iterations)...")
        times = []

        for i in range(num_iters):
            start = time.time()
            with torch.no_grad():
                pred = model.infer_image(rgb, depth, input_size=518)

            # Synchronize if using CUDA
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}/{num_iters}: {elapsed*1000:.2f}ms")

        times = np.array(times)

        # Results
        result = {
            "mean": times.mean(),
            "std": times.std(),
            "min": times.min(),
            "max": times.max(),
            "output_shape": pred.shape,
            "success": True,
        }

        print("\n✓ Test passed!")
        print(f"  Mean: {result['mean']*1000:.2f}ms ± {result['std']*1000:.2f}ms")
        print(f"  Min:  {result['min']*1000:.2f}ms")
        print(f"  Max:  {result['max']*1000:.2f}ms")
        print(f"  Output shape: {result['output_shape']}")

        return result

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_all_attention_backends(device="auto"):
    """Test all attention backends on the specified device."""
    print(f"\n{'#'*60}")
    print(f"# Testing All Attention Backends on {device}")
    print(f"{'#'*60}")

    backends = ["torch", "manual"]

    # Add xformers if on CUDA
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        try:
            import xformers

            backends.insert(0, "xformers")
        except ImportError:
            print("xformers not available, skipping xformers tests")

    results = {}

    for backend in backends:
        config = OptimizationConfig(
            device=device,
            attention_backend=backend,
            use_compile=False,  # Disable for testing
            mixed_precision="fp32",  # Use FP32 for fair comparison
        )

        result = test_configuration(config=config, num_warmup=2, num_iters=5)
        results[backend] = result

    # Summary
    print(f"\n{'='*60}")
    print("ATTENTION BACKEND COMPARISON")
    print(f"{'='*60}")
    print(f"{'Backend':<15} {'Mean (ms)':<12} {'Success':<10}")
    print("-" * 60)

    for backend, result in results.items():
        if result:
            print(f"{backend:<15} {result['mean']*1000:>10.2f}   {'✓':<10}")
        else:
            print(f"{backend:<15} {'N/A':>10}   {'✗':<10}")


def test_device_specific_optimizations():
    """Test device-specific optimizations."""
    device = "auto"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{'#'*60}")
    print(f"# Testing Device-Specific Optimizations: {device}")
    print(f"{'#'*60}")

    configs = []

    if device == "cuda":
        # Test CUDA optimizations
        configs = [
            (
                "Baseline (FP32)",
                OptimizationConfig(
                    device=device,
                    attention_backend="torch",
                    use_compile=False,
                    mixed_precision="fp32",
                    use_channels_last=False,
                ),
            ),
            (
                "FP16",
                OptimizationConfig(
                    device=device,
                    attention_backend="torch",
                    use_compile=False,
                    mixed_precision="fp16",
                    use_channels_last=False,
                ),
            ),
            (
                "Channels Last",
                OptimizationConfig(
                    device=device,
                    attention_backend="torch",
                    use_compile=False,
                    mixed_precision="fp32",
                    use_channels_last=True,
                ),
            ),
            (
                "All Optimizations",
                OptimizationConfig(
                    device=device,
                    attention_backend="auto",
                    use_compile=False,  # Compile can be flaky in tests
                    mixed_precision="fp16",
                    use_channels_last=True,
                ),
            ),
        ]
    elif device == "mps":
        # Test MPS optimizations
        configs = [
            (
                "Baseline",
                OptimizationConfig(
                    device=device,
                    attention_backend="torch",
                    use_channels_last=False,
                    mixed_precision="fp32",
                ),
            ),
            (
                "Manual Attention",
                OptimizationConfig(
                    device=device,
                    attention_backend="manual",
                    use_channels_last=False,
                    mixed_precision="fp32",
                ),
            ),
            (
                "Channels Last",
                OptimizationConfig(
                    device=device,
                    attention_backend="manual",
                    use_channels_last=True,
                    mixed_precision="fp32",
                ),
            ),
            (
                "All Optimizations",
                OptimizationConfig(
                    device=device,
                    attention_backend="manual",
                    use_channels_last=True,
                    mixed_precision="fp32",
                    fuse_depth_encoder=True,
                ),
            ),
        ]
    else:  # CPU
        configs = [
            (
                "Baseline",
                OptimizationConfig(device=device, use_channels_last=False, mixed_precision="fp32"),
            ),
            (
                "Channels Last",
                OptimizationConfig(device=device, use_channels_last=True, mixed_precision="fp32"),
            ),
            (
                "All Optimizations",
                OptimizationConfig(
                    device=device,
                    use_channels_last=True,
                    mixed_precision="fp32",
                    fuse_depth_encoder=True,
                ),
            ),
        ]

    results = {}
    for name, config in configs:
        print(f"\nTesting: {name}")
        result = test_configuration(config=config, num_warmup=1, num_iters=3)
        results[name] = result

    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Configuration':<25} {'Mean (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    baseline_time = results.get("Baseline", {}).get("mean")

    for name, result in results.items():
        if result:
            speedup = ""
            if baseline_time and name != "Baseline":
                speedup = f"{baseline_time / result['mean']:.2f}x"
            print(f"{name:<25} {result['mean']*1000:>10.2f}   {speedup:<10}")
        else:
            print(f"{name:<25} {'FAILED':>10}   {'-':<10}")


def main():
    """Main test runner."""
    print("CDM Optimization Tests")
    print("=" * 60)

    # Print system info
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")

    try:
        import xformers

        print("  xformers available: True")
    except ImportError:
        print("  xformers available: False")

    # Run tests
    try:
        # Test 1: All attention backends
        test_all_attention_backends()

        # Test 2: Device-specific optimizations
        test_device_specific_optimizations()

        print(f"\n{'='*60}")
        print("All tests completed!")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nTests failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
