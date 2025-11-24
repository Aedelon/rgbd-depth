#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Basic import and smoke tests for camera-depth-models."""

import pytest


def test_import_main_modules():
    """Test that main modules can be imported."""
    from rgbddepth import OptimizationConfig, RGBDDepth, get_optimal_config

    assert RGBDDepth is not None
    assert OptimizationConfig is not None
    assert get_optimal_config is not None


def test_import_attention():
    """Test attention module imports."""
    from rgbddepth.attention import AdaptiveCrossAttention, create_cross_attention

    assert AdaptiveCrossAttention is not None
    assert create_cross_attention is not None


def test_optimization_config_auto():
    """Test OptimizationConfig with auto device."""
    from rgbddepth import OptimizationConfig

    config = OptimizationConfig(device="auto")
    assert config.device in ["cuda", "mps", "cpu"]
    assert config.attention_backend in ["xformers", "torch", "manual"]
    assert config.mixed_precision in ["fp16", "bf16", "fp32"]


def test_optimization_config_cpu():
    """Test OptimizationConfig for CPU."""
    from rgbddepth import OptimizationConfig

    config = OptimizationConfig(device="cpu")
    assert config.device == "cpu"
    assert config.mixed_precision == "fp32"
    assert config.use_compile is False


def test_model_creation():
    """Test that model can be instantiated."""
    from rgbddepth import OptimizationConfig, RGBDDepth

    config = OptimizationConfig(device="cpu")
    model = RGBDDepth(
        encoder="vits",  # Use smallest model for testing
        features=64,
        out_channels=[48, 96, 192, 384],
        config=config,
    )

    assert model is not None
    assert hasattr(model, "forward")
    assert hasattr(model, "infer_image")


def test_model_forward_shape():
    """Test model forward pass with dummy data."""
    import torch

    from rgbddepth import OptimizationConfig, RGBDDepth

    config = OptimizationConfig(device="cpu")
    model = RGBDDepth(encoder="vits", features=64, out_channels=[48, 96, 192, 384], config=config)
    model.eval()

    # Create dummy inputs
    batch_size = 1
    height, width = 224, 224
    rgb = torch.randn(batch_size, 3, height, width)
    depth = torch.randn(batch_size, 1, height, width)

    # Concatenate RGB and depth as model expects (B, 4, H, W)
    inputs = torch.cat([rgb, depth], dim=1)

    with torch.no_grad():
        output = model(inputs)

    # Output shape is (B, H, W) after squeeze(1)
    assert output.shape[0] == batch_size
    assert output.shape[1] == height
    assert output.shape[2] == width


def test_get_optimal_config():
    """Test get_optimal_config function."""
    from rgbddepth import get_optimal_config

    config = get_optimal_config()
    assert config is not None
    assert config.device is not None


def test_cli_imports():
    """Test that CLI modules can be imported."""
    from rgbddepth.cli import main_download, main_infer

    assert main_download is not None
    assert main_infer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
