#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive model tests for rgbd-depth (CPU-only, no checkpoints needed)."""

import numpy as np
import pytest
import torch

from rgbddepth import RGBDDepth
from rgbddepth.flexible_attention import FlexibleCrossAttention


class TestModelArchitecture:
    """Test model architecture and configurations."""

    @pytest.mark.parametrize("encoder", ["vits", "vitb", "vitl"])
    def test_different_encoders(self, encoder):
        """Test model creation with different encoder sizes."""
        model = RGBDDepth(encoder=encoder, features=256, use_xformers=False)
        assert model is not None
        assert hasattr(model, "pretrained")

    @pytest.mark.parametrize("features", [128, 256, 512])
    def test_different_feature_dims(self, features):
        """Test model with different feature dimensions."""
        model = RGBDDepth(encoder="vitl", features=features, use_xformers=False)
        assert model is not None

    def test_custom_out_channels(self):
        """Test model with custom output channels."""
        out_channels = [128, 256, 512, 512]
        model = RGBDDepth(
            encoder="vitl", features=256, out_channels=out_channels, use_xformers=False
        )
        assert model is not None

    def test_model_eval_mode(self):
        """Test model can be set to eval mode."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()
        assert not model.training


class TestForwardPass:
    """Test forward pass with different inputs."""

    def test_forward_basic(self):
        """Test basic forward pass with 4-channel input."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        # 4-channel input: RGB + depth
        x = torch.randn(1, 4, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 224, 224)

    @pytest.mark.parametrize("size", [224, 336, 518])
    def test_different_input_sizes(self, size):
        """Test forward pass with different input resolutions."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        x = torch.randn(1, 4, size, size)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, size, size)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, batch_size):
        """Test forward pass with different batch sizes."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        x = torch.randn(batch_size, 4, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 224, 224)

    def test_output_is_finite(self):
        """Test that output contains finite values (no NaN/Inf)."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        x = torch.randn(1, 4, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert torch.all(torch.isfinite(output))


class TestInferImage:
    """Test infer_image convenience method."""

    def test_infer_image_numpy(self):
        """Test infer_image with numpy arrays."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        # Create numpy inputs
        rgb = np.random.rand(224, 224, 3).astype(np.float32)
        depth = np.random.rand(224, 224).astype(np.float32)

        output = model.infer_image(rgb, depth, input_size=224)

        assert isinstance(output, np.ndarray)
        assert output.shape == (224, 224)
        assert np.all(np.isfinite(output))

    @pytest.mark.parametrize("input_size", [224, 336, 518])
    def test_infer_image_different_sizes(self, input_size):
        """Test infer_image with different input sizes."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        rgb = np.random.rand(input_size, input_size, 3).astype(np.float32)
        depth = np.random.rand(input_size, input_size).astype(np.float32)

        output = model.infer_image(rgb, depth, input_size=input_size)

        assert output.shape == (input_size, input_size)

    def test_infer_image_resize(self):
        """Test infer_image handles resize correctly."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model.eval()

        # Different input vs processing size
        rgb = np.random.rand(480, 640, 3).astype(np.float32)
        depth = np.random.rand(480, 640).astype(np.float32)

        output = model.infer_image(rgb, depth, input_size=518)

        # Output should match original size
        assert output.shape == (480, 640)


class TestFlexibleAttention:
    """Test FlexibleCrossAttention module."""

    def test_attention_creation(self):
        """Test FlexibleCrossAttention can be created."""
        attn = FlexibleCrossAttention(256, 8, use_xformers=False)
        assert attn is not None
        assert attn.embed_dim == 256
        assert attn.num_heads == 8

    def test_attention_forward_shape(self):
        """Test attention forward pass shape."""
        attn = FlexibleCrossAttention(256, 8, use_xformers=False)
        attn.eval()

        q = k = v = torch.randn(2, 100, 256)

        with torch.no_grad():
            out, _ = attn(q, k, v)

        assert out.shape == (2, 100, 256)

    def test_attention_without_xformers(self):
        """Test attention without xFormers (SDPA fallback)."""
        attn = FlexibleCrossAttention(256, 8, use_xformers=False)
        attn.eval()

        x = torch.randn(2, 100, 256)

        with torch.no_grad():
            out, _ = attn(x, x, x)

        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    def test_attention_with_xformers_flag(self):
        """Test attention with xFormers flag (may fallback to SDPA)."""
        # This will attempt xFormers but fallback if not available
        attn = FlexibleCrossAttention(256, 8, use_xformers=True)
        attn.eval()

        x = torch.randn(2, 100, 256)

        with torch.no_grad():
            out, _ = attn(x, x, x)

        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    def test_attention_different_heads(self, num_heads):
        """Test attention with different number of heads."""
        embed_dim = 256
        attn = FlexibleCrossAttention(embed_dim, num_heads, use_xformers=False)
        attn.eval()

        x = torch.randn(2, 100, embed_dim)

        with torch.no_grad():
            out, _ = attn(x, x, x)

        assert out.shape == x.shape


class TestDepthPreprocessing:
    """Test depth preprocessing utilities."""

    def test_inverse_depth_conversion(self):
        """Test inverse depth (similarity depth) conversion."""
        # Simulate depth preprocessing from app.py
        depth = np.random.uniform(0.5, 10.0, size=(100, 100)).astype(np.float32)

        # Create inverse depth
        simi_depth = np.zeros_like(depth)
        valid_mask = depth > 0
        simi_depth[valid_mask] = 1.0 / depth[valid_mask]

        # Check inverse relationship
        assert np.all(simi_depth[valid_mask] > 0)
        assert np.all(simi_depth[valid_mask] <= 2.0)  # max when depth=0.5

    def test_depth_normalization(self):
        """Test depth normalization."""
        # Raw depth in arbitrary units
        depth_raw = np.random.uniform(0, 5000, size=(100, 100)).astype(np.float32)
        depth_scale = 1000.0
        max_depth = 25.0

        # Normalize
        depth_normalized = depth_raw / depth_scale
        depth_normalized[depth_normalized > max_depth] = 0.0

        assert np.all(depth_normalized >= 0)
        assert np.all(depth_normalized <= max_depth)

    def test_inverse_depth_back_conversion(self):
        """Test converting inverse depth back to depth."""
        # Simulate model output (inverse depth)
        inverse_depth = np.random.uniform(0.01, 2.0, size=(100, 100)).astype(np.float32)

        # Convert back to depth
        depth = np.where(inverse_depth > 1e-8, 1.0 / inverse_depth, 0.0)

        assert np.all(depth >= 0)
        assert np.all(depth[depth > 0] <= 100.0)  # max when inverse=0.01


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_encoder(self):
        """Test that invalid encoder raises error."""
        with pytest.raises((ValueError, AssertionError, KeyError)):
            RGBDDepth(encoder="invalid_encoder", use_xformers=False)

    def test_mismatched_out_channels(self):
        """Test that mismatched out_channels length raises IndexError."""
        # Should raise IndexError when out_channels list is too short
        with pytest.raises(IndexError):
            RGBDDepth(encoder="vitl", features=256, out_channels=[256, 512], use_xformers=False)


class TestDeviceCompatibility:
    """Test device compatibility (CPU, CUDA, MPS)."""

    def test_cpu_inference(self):
        """Test model runs on CPU."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model = model.to("cpu").eval()

        x = torch.randn(1, 4, 224, 224, device="cpu")

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self):
        """Test model runs on CUDA if available."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model = model.to("cuda").eval()

        x = torch.randn(1, 4, 224, 224, device="cuda")

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_inference(self):
        """Test model runs on MPS if available."""
        model = RGBDDepth(encoder="vitl", use_xformers=False)
        model = model.to("mps").eval()

        x = torch.randn(1, 4, 224, 224, device="mps")

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "mps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
