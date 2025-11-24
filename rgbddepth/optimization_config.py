#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Device-specific optimization configuration for CDM models.
Provides automatic device detection and manual override capabilities.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Literal
import warnings


@dataclass
class OptimizationConfig:
    """Configuration for device-specific optimizations.

    Args:
        device: Target device ('cuda', 'cpu', 'mps', or 'auto')
        attention_backend: Attention implementation to use
            - 'auto': Automatically select best for device
            - 'xformers': Use xformers (GPU only, fastest)
            - 'torch': Use torch.nn.MultiheadAttention (all devices)
            - 'manual': Custom memory-efficient implementation (MPS recommended)
        use_compile: Enable torch.compile (GPU only, PyTorch 2.0+)
        use_channels_last: Use channels_last memory format (CPU/MPS benefit most)
        mixed_precision: Enable mixed precision training/inference
            - 'auto': FP16 on CUDA, FP32 on CPU/MPS
            - 'fp16': Force FP16
            - 'bf16': Force BF16 (requires Ampere+ GPU or recent CPU)
            - 'fp32': Force FP32
        fuse_depth_encoder: Use single encoder for RGB+depth (saves memory, better for CPU/MPS)
        interpolation_mode: Mode for spatial interpolation
            - 'bicubic': Highest quality, slowest (GPU handles well)
            - 'bilinear': Good balance (recommended for CPU/MPS)
            - 'nearest': Fastest, lowest quality
    """

    device: str = "auto"
    attention_backend: Literal["auto", "xformers", "torch", "manual"] = "auto"
    use_compile: Optional[bool] = None  # None = auto-detect
    use_channels_last: Optional[bool] = None  # None = auto-detect
    mixed_precision: Literal["auto", "fp16", "bf16", "fp32"] = "auto"
    fuse_depth_encoder: Optional[bool] = None  # None = auto-detect
    interpolation_mode: Literal["bicubic", "bilinear", "nearest"] = "bilinear"

    def __post_init__(self):
        """Auto-configure based on device if not explicitly set."""
        # Resolve device
        if self.device == "auto":
            self.device = self._detect_device()

        # Auto-configure based on device
        device_type = self.device.split(":")[0]  # Handle cuda:0 format

        # Attention backend
        if self.attention_backend == "auto":
            self.attention_backend = self._select_attention_backend(device_type)

        # Validate attention backend
        self._validate_attention_backend(device_type)

        # Torch compile
        if self.use_compile is None:
            self.use_compile = self._should_use_compile(device_type)

        # Channels last
        if self.use_channels_last is None:
            self.use_channels_last = self._should_use_channels_last(device_type)

        # Fuse depth encoder
        if self.fuse_depth_encoder is None:
            self.fuse_depth_encoder = self._should_fuse_encoder(device_type)

        # Mixed precision
        self.dtype = self._resolve_dtype(device_type)

        # Interpolation mode - use bicubic only on CUDA
        if self.interpolation_mode == "bicubic" and device_type != "cuda":
            warnings.warn(
                f"Bicubic interpolation is slow on {device_type}, "
                f"consider using 'bilinear' for better performance"
            )

    def _detect_device(self) -> str:
        """Automatically detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _select_attention_backend(self, device_type: str) -> str:
        """Select optimal attention backend for device."""
        if device_type == "cuda":
            # Try xformers first, fall back to torch
            try:
                import xformers
                return "xformers"
            except ImportError:
                return "torch"
        elif device_type == "mps":
            # MPS has poor MultiheadAttention performance
            return "manual"
        else:  # CPU
            return "torch"

    def _validate_attention_backend(self, device_type: str):
        """Validate attention backend is compatible with device."""
        if self.attention_backend == "xformers":
            if device_type != "cuda":
                raise ValueError(
                    f"xformers backend only supports CUDA, not {device_type}. "
                    f"Use 'torch' or 'manual' backend instead."
                )
            try:
                import xformers
            except ImportError:
                raise ImportError(
                    "xformers is not installed. Install with: "
                    "pip install xformers\n"
                    "Or use attention_backend='torch' or 'auto'"
                )

    def _should_use_compile(self, device_type: str) -> bool:
        """Determine if torch.compile should be used."""
        # Only beneficial on CUDA with PyTorch 2.0+ and triton installed
        if device_type != "cuda":
            return False

        try:
            import triton  # noqa: F401
        except Exception:
            return False

        # Check PyTorch version
        torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if torch_version < (2, 0):
            return False

        return True

    def _should_use_channels_last(self, device_type: str) -> bool:
        """Determine if channels_last memory format should be used."""
        # Benefits CPU most. On MPS, limited benefit as it's only applied to decoder
        # (ViT encoders have .view() incompatibilities with channels_last)
        # On CUDA, slight benefit for convolutions
        return device_type in ["cpu", "cuda"]

    def _should_fuse_encoder(self, device_type: str) -> bool:
        """Determine if RGB and depth encoders should be fused."""
        # Fusing saves memory, beneficial for CPU/MPS
        # GPU can handle separate encoders with more parallelism
        return device_type in ["cpu", "mps"]

    def _resolve_dtype(self, device_type: str) -> torch.dtype:
        """Resolve mixed precision dtype."""
        if self.mixed_precision == "fp32":
            return torch.float32
        elif self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == "bf16":
            if not torch.cuda.is_bf16_supported() and device_type == "cuda":
                warnings.warn(
                    "BF16 not supported on this GPU, falling back to FP32"
                )
                return torch.float32
            return torch.bfloat16
        else:  # auto
            if device_type == "cuda":
                # Use BF16 on Ampere+, FP16 otherwise
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            elif device_type == "mps":
                # Use FP32 on MPS by default
                # While FP16 saves memory (~2x), it can be slower on MPS for some models
                # due to conversion overhead and lack of dedicated FP16 hardware
                # Users can manually enable FP16 with mixed_precision='fp16' for memory savings
                return torch.float32
            else:
                # CPU: use FP32 only (FP16 is slower on CPU)
                return torch.float32

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "=== CDM Optimization Configuration ===",
            f"Device: {self.device}",
            f"Attention Backend: {self.attention_backend}",
            f"Torch Compile: {'✓' if self.use_compile else '✗'}",
            f"Channels Last: {'✓' if self.use_channels_last else '✗'}",
            f"Mixed Precision: {self.mixed_precision} ({self.dtype})",
            f"Fuse Depth Encoder: {'✓' if self.fuse_depth_encoder else '✗'}",
            f"Interpolation Mode: {self.interpolation_mode}",
            "=" * 38
        ]
        return "\n".join(lines)


def get_optimal_config(device: str = "auto") -> OptimizationConfig:
    """Get optimal configuration for the specified device.

    Args:
        device: Target device ('cuda', 'cpu', 'mps', or 'auto')

    Returns:
        OptimizationConfig with optimal settings for the device
    """
    return OptimizationConfig(device=device)


def get_config_for_inference(
    device: str = "auto",
    prefer_quality: bool = False
) -> OptimizationConfig:
    """Get configuration optimized for inference.

    Args:
        device: Target device
        prefer_quality: If True, prefer quality over speed
            (e.g., bicubic interpolation, fp32 precision)

    Returns:
        OptimizationConfig optimized for inference
    """
    config = OptimizationConfig(device=device)

    if prefer_quality:
        config.interpolation_mode = "bicubic"
        config.mixed_precision = "fp32"

    return config


def get_config_for_training(device: str = "auto") -> OptimizationConfig:
    """Get configuration optimized for training.

    Args:
        device: Target device

    Returns:
        OptimizationConfig optimized for training
    """
    config = OptimizationConfig(device=device)

    # Training-specific adjustments
    # Disable compile during training for better debugging
    config.use_compile = False

    # Use separate encoders for better gradient flow
    config.fuse_depth_encoder = False

    return config
