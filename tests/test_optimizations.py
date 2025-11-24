#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Test optimization configurations."""

import pytest
import torch
from rgbddepth import OptimizationConfig


class TestOptimizationConfig:
    """Test suite for OptimizationConfig."""

    def test_auto_device_detection(self):
        """Test automatic device detection."""
        config = OptimizationConfig(device="auto")

        if torch.cuda.is_available():
            assert config.device == "cuda"
        elif torch.backends.mps.is_available():
            assert config.device == "mps"
        else:
            assert config.device == "cpu"

    def test_cuda_config(self):
        """Test CUDA configuration."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = OptimizationConfig(device="cuda")
        assert config.device == "cuda"
        assert config.attention_backend in ["xformers", "torch"]
        assert config.mixed_precision in ["fp16", "bf16"]

    def test_mps_config(self):
        """Test MPS configuration."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        config = OptimizationConfig(device="mps")
        assert config.device == "mps"
        assert config.attention_backend == "manual"
        assert config.mixed_precision == "fp32"

    def test_cpu_config(self):
        """Test CPU configuration."""
        config = OptimizationConfig(device="cpu")
        assert config.device == "cpu"
        assert config.attention_backend == "torch"
        assert config.mixed_precision == "fp32"
        assert config.use_compile is False

    def test_manual_override(self):
        """Test manual configuration override."""
        config = OptimizationConfig(
            device="cpu",
            attention_backend="manual",
            use_compile=False,
            mixed_precision="fp32"
        )

        assert config.device == "cpu"
        assert config.attention_backend == "manual"
        assert config.use_compile is False
        assert config.mixed_precision == "fp32"

    def test_summary(self):
        """Test configuration summary."""
        config = OptimizationConfig(device="cpu")
        summary = config.summary()

        assert "Device" in summary
        assert "cpu" in summary
        assert "Attention" in summary

    def test_invalid_device(self):
        """Test invalid device raises error."""
        with pytest.raises(ValueError):
            OptimizationConfig(device="invalid")

    def test_invalid_precision(self):
        """Test invalid precision raises error."""
        with pytest.raises(ValueError):
            OptimizationConfig(device="cpu", mixed_precision="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
