"""RGBD Depth - Optimized RGB-D depth refinement using Vision Transformers.

This package provides optimized depth refinement for RGB-D cameras with support
for CUDA (xFormers), MPS (Apple Silicon), and CPU devices.
"""

__version__ = "1.0.2"

from .dinov2 import DinoVisionTransformer
from .dpt import RGBDDepth

__all__ = ["RGBDDepth", "DinoVisionTransformer", "__version__"]
