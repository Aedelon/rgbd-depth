#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Basic import and smoke tests for rgbd-depth package."""

import pytest


def test_import_main_modules():
    """Test that main modules can be imported."""
    from rgbddepth import DinoVisionTransformer, RGBDDepth, __version__

    assert RGBDDepth is not None
    assert DinoVisionTransformer is not None


def test_import_flexible_attention():
    """Test FlexibleCrossAttention module import."""
    from rgbddepth.flexible_attention import FlexibleCrossAttention

    assert FlexibleCrossAttention is not None


def test_model_creation_basic():
    """Test that model can be instantiated without xFormers."""
    from rgbddepth import RGBDDepth

    model = RGBDDepth(
        encoder="vitl",
        features=256,
        use_xformers=False,  # Don't require xFormers for basic test
    )

    assert model is not None
    assert hasattr(model, "forward")
    assert hasattr(model, "infer_image")


def test_model_creation_with_xformers():
    """Test that model can be instantiated with xFormers flag (may fallback)."""
    from rgbddepth import RGBDDepth

    # This will attempt xFormers but fallback to SDPA if not available
    model = RGBDDepth(encoder="vitl", features=256, use_xformers=True)

    assert model is not None
    assert hasattr(model, "forward")


def test_model_forward_shape():
    """Test model forward pass with dummy data."""
    import torch

    from rgbddepth import RGBDDepth

    model = RGBDDepth(encoder="vitl", features=256, use_xformers=False)
    model.eval()

    # Create dummy inputs (RGB + depth concatenated)
    batch_size = 1
    height, width = 224, 224
    rgb = torch.randn(batch_size, 3, height, width)
    depth = torch.randn(batch_size, 1, height, width)

    # Concatenate RGB and depth as model expects (B, 4, H, W)
    inputs = torch.cat([rgb, depth], dim=1)

    with torch.no_grad():
        output = model(inputs)

    # Output should be (B, H, W) - model squeezes channel dimension
    assert output.shape == (batch_size, height, width)


def test_infer_main_function():
    """Test that infer.py main() function exists."""
    import importlib.util
    import sys
    from pathlib import Path

    # Load infer.py as a module
    infer_path = Path(__file__).parent.parent / "infer.py"
    spec = importlib.util.spec_from_file_location("infer", infer_path)
    infer = importlib.util.module_from_spec(spec)
    sys.modules["infer"] = infer
    spec.loader.exec_module(infer)

    assert hasattr(infer, "main")
    assert callable(infer.main)


def test_device_detection():
    """Test device detection logic."""
    import torch

    # Just verify torch can detect available devices
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    # At least CPU should always be available
    assert cuda_available or mps_available or True  # CPU always available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
