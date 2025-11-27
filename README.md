---
title: rgbd-depth
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
---

# Camera Depth Models (CDM)

Optimized Python package for RGB-D depth refinement using Vision Transformer encoders. This implementation is aligned with the [ByteDance CDM reference implementation](https://github.com/bytedance/camera-depth-models) with additional performance optimizations for CUDA, MPS (Apple Silicon), and CPU.

[![Tests](https://github.com/Aedelon/rgbd-depth/actions/workflows/test.yml/badge.svg)](https://github.com/Aedelon/rgbd-depth/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/rgbd-depth.svg)](https://pypi.org/project/rgbd-depth/)
[![PyPI downloads](https://img.shields.io/pypi/dm/rgbd-depth.svg)](https://pypi.org/project/rgbd-depth/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Aedelon/rgbd-depth)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎮 Try it Online

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Aedelon/rgbd-depth)

Try **rgbd-depth** directly in your browser with our interactive Gradio demo—no installation required. Upload your images and refine depth maps instantly.

**Available on Hugging Face Spaces:** Upload your RGB and depth images, adjust parameters (camera model, precision, resolution), and get refined depth maps instantly. Models are automatically downloaded from Hugging Face Hub on first use.

## Overview

Camera Depth Models (CDMs) are sensor-specific depth models trained to produce clean, simulation-like depth maps from noisy real-world depth camera data. By bridging the visual gap between simulation and reality through depth perception, CDMs enable robotic policies trained purely in simulation to transfer directly to real robots.

**Original work by ByteDance Research.** This package provides an optimized implementation with:
- ✅ **Pixel-perfect alignment** with reference implementation (verified: 0 pixel difference)
- ⚡ **Device-specific optimizations**: xFormers (CUDA), SDPA fallback, torch.compile
- 🎯 **Mixed precision support**: FP16 (CUDA/MPS), BF16 (CUDA)
- 🔧 **Better CLI**: Device selection, optimization control, precision modes
- 📦 **Easy installation**: Single `pip install` command

## Why This Package?

This is an **optimized, production-ready** version of ByteDance's Camera Depth Models with several improvements:

| Feature | ByteDance Original | This Package |
|---------|-------------------|--------------|
| **Installation** | Manual setup | `pip install rgbd-depth` |
| **CUDA Optimization** | Basic | xFormers (~8% faster) + torch.compile |
| **Apple Silicon (MPS)** | Not optimized | Native support with fallbacks |
| **Mixed Precision** | Manual | Automatic FP16/BF16 with `--precision` flag |
| **CLI** | Basic | Enhanced with device selection, optimization control |
| **Documentation** | Minimal | Comprehensive guides (README + OPTIMIZATION.md) |
| **Testing** | None | CI/CD with automated tests |
| **PyPI Package** | No | ✅ Yes (`rgbd-depth`) |

**Choose this package if you want:**
- 🚀 Faster inference on CUDA (xFormers) or Apple Silicon (MPS)
- 🎯 Easy mixed precision (FP16/BF16) without code changes
- 📦 Simple installation via PyPI
- 🔧 Production-ready CLI with device/precision control
- ✅ Maintained with CI/CD and tests

### Key Features

- **Metric Depth Estimation**: Produces accurate absolute depth measurements in meters
- **Multi-Camera Support**: Optimized models for various depth sensors (RealSense D405/D435/L515, ZED 2i, Azure Kinect)
- **Performance Optimizations**: ~8% faster on CUDA with xFormers, automatic backend selection
- **Mixed Precision**: FP16/BF16 support for faster inference on compatible hardware
- **Sim-to-Real Ready**: Generates simulation-quality depth from real camera data

## Architecture

CDM uses a dual-branch Vision Transformer architecture:
- **RGB Branch**: Extracts semantic information from RGB images
- **Depth Branch**: Processes noisy depth sensor data
- **Cross-Attention Fusion**: Combines RGB semantics with depth scale information
- **DPT Decoder**: Produces final metric depth estimation

Supported ViT encoder sizes:
- `vits`: Small (64 features, 384 output channels)
- `vitb`: Base (128 features, 768 output channels)
- `vitl`: Large (256 features, 1024 output channels)
- `vitg`: Giant (384 features, 1536 output channels)

All pretrained models we provide are based on `vitl`.

## 🌐 Hugging Face Spaces Demo

The easiest way to try rgbd-depth is via **Hugging Face Spaces**—completely free, no installation needed:

1. **Open the [interactive demo](https://huggingface.co/spaces/Aedelon/rgbd-depth)**
2. **Upload** an RGB image and a depth map (PNG or JPG)
3. **Configure** camera model, precision, and visualization options
4. **Click "Refine Depth"** and download the result

**What happens:**
- Models are auto-downloaded from Hugging Face Hub on first use
- Runs on free CPU hardware (inference: ~10-30s)
- GPU hardware available for faster processing (~2-5s)
- All computations are done server-side—your images stay private

**Limitations (HF Spaces CPU):**
- No xFormers optimization (CUDA-only)
- Inference slower than local GPU
- Perfect for testing and prototyping

For production workflows or faster inference, use the local installation below.

> **📌 Note:** This README is optimized for [GitHub](https://github.com/Aedelon/rgbd-depth), [PyPI](https://pypi.org/project/rgbd-depth/), and [Hugging Face Spaces](https://huggingface.co/spaces/Aedelon/rgbd-depth). The YAML metadata (top of file) is auto-detected by HF Spaces and not displayed.

## Installation

### From PyPI (recommended)

**Basic installation (core dependencies only):**
```bash
pip install rgbd-depth
```

**Installation with extras:**
```bash
# With CUDA optimizations (xFormers, ~8% faster)
pip install rgbd-depth[xformers]

# With Gradio demo interface
pip install rgbd-depth[demo]

# With HuggingFace Hub model downloads
pip install rgbd-depth[download]

# With development tools (pytest, black, ruff, etc.)
pip install rgbd-depth[dev]

# Install everything (all extras)
pip install rgbd-depth[all]
```

**Development installation (editable):**
```bash
git clone https://github.com/Aedelon/rgbd-depth.git
cd rgbd-depth
pip install -e ".[dev]"  # or uv sync --extra dev
```

**Requirements:**
- Python 3.10+ (Python 3.8-3.9 support dropped in v1.0.2+)
- PyTorch 2.0+ with appropriate CUDA/MPS support
- OpenCV, NumPy, Pillow

## Quick Start

### Easiest: No Installation (HF Spaces)

👉 **[Open interactive demo in your browser](https://huggingface.co/spaces/Aedelon/rgbd-depth)** ← Start here!

### Local Installation

After `pip install rgbd-depth`:

```bash
# CUDA (optimizations auto-enabled, FP16 for best speed)
python infer.py --input rgb.png --depth depth.png --precision fp16

# Apple Silicon (MPS)
python infer.py --input rgb.png --depth depth.png --device mps

# CPU (FP32 only)
python infer.py --input rgb.png --depth depth.png --device cpu
```

> Example images are provided in `example_data/`. Pre-trained models can be downloaded from [Hugging Face](https://huggingface.co/collections/depth-anything/camera-depth-models-68b521181dedd223f4b020db).

## Usage

### Command Line Interface

**Basic inference:**
```bash
python infer.py \
    --input /path/to/rgb.png \
    --depth /path/to/depth.png \
    --output refined_depth.png
```

**CUDA with optimizations (default):**
```bash
# FP32 (best accuracy)
python infer.py --input rgb.png --depth depth.png

# FP16 (best speed, ~2× faster)
python infer.py --input rgb.png --depth depth.png --precision fp16

# BF16 (best stability)
python infer.py --input rgb.png --depth depth.png --precision bf16

# Disable optimizations (debugging)
python infer.py --input rgb.png --depth depth.png --no-optimize
```

**Apple Silicon (MPS):**
```bash
# FP32 (default)
python infer.py --input rgb.png --depth depth.png --device mps

# FP16 (faster)
python infer.py --input rgb.png --depth depth.png --device mps --precision fp16
```

**CPU:**
```bash
# FP32 only (FP16 not recommended on CPU)
python infer.py --input rgb.png --depth depth.png --device cpu
```

### Command Line Arguments

**Required:**
- `--input`: Path to RGB input image (JPG/PNG)
- `--depth`: Path to depth input image (PNG, 16-bit or 32-bit)

**Optional:**
- `--output`: Output visualization path (default: `output.png`)
- `--device`: Device to use: `auto`, `cuda`, `mps`, `cpu` (default: `auto`)
- `--precision`: Precision mode: `fp32`, `fp16`, `bf16` (default: `fp32`)
- `--no-optimize`: Disable optimizations on CUDA (for debugging)
- `--encoder`: Model size: `vits`, `vitb`, `vitl`, `vitg` (default: `vitl`)
- `--input-size`: Input resolution for inference (default: 518)
- `--depth-scale`: Scale factor for depth values (default: 1000.0)
- `--max-depth`: Maximum valid depth in meters (default: 6.0)

### Python API

```python
import torch
from rgbddepth.dpt import RGBDDepth
import cv2
import numpy as np

# Load model with optimizations
model = RGBDDepth(encoder='vitl', features=256, use_xformers=True)
model.load_state_dict(torch.load('model.pth'))
model.eval()
model = model.to('cuda')  # or 'mps', 'cpu'

# Optional: compile for extra speed on CUDA
model = torch.compile(model)

# Load images
rgb = cv2.imread('rgb.jpg')[:, :, ::-1]  # BGR to RGB
depth = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED) / 1000.0  # Convert to meters

# Create similarity depth (inverse depth)
simi_depth = np.zeros_like(depth)
simi_depth[depth > 0] = 1 / depth[depth > 0]

# Run inference with mixed precision
with torch.amp.autocast('cuda', dtype=torch.float16):
    pred_depth = model.infer_image(rgb, simi_depth, input_size=518)
```

## Model Training

CDMs are trained on synthetic datasets generated using camera-specific noise models:

1. **Noise Model Training**: Learn hole and value noise patterns from real camera data
2. **Synthetic Data Generation**: Apply learned noise to clean simulation depth
3. **CDM Training**: Train depth estimation model on synthetic noisy data

Training datasets: HyperSim, DREDS, HISS, IRS (280,000+ images total)

## Supported Cameras

We currently provide pre-trained models available for:
- Intel RealSense D405/D435/L515
- Stereolabs ZED 2i (2 modes: Quality, Neural)
- Microsoft Azure Kinect

## File Structure

```
rgbd-depth/
├── app.py                      # Gradio web demo for HuggingFace Spaces
├── infer.py                    # CLI inference script (main entry point)
├── pyproject.toml              # Modern package config (PEP 621, replaces setup.py)
├── setup.py                    # Legacy setuptools build script
├── requirements.txt            # Minimal deps for HuggingFace Spaces
├── uv.lock                     # UV package manager lock file
├── LICENSE                     # Apache 2.0 license
├── README.md                   # This file (GitHub/PyPI/HF Spaces unified)
├── OPTIMIZATION.md             # Performance benchmarks and optimization guide
├── CHANGELOG.md                # Version history and release notes
└── VIRAL_STRATEGY.md           # GitHub/PyPI marketing strategy
│
├── rgbddepth/                  # Main Python package
│   ├── __init__.py             # Public API exports (RGBDDepth, DinoVisionTransformer, __version__)
│   ├── dpt.py                  # RGBDDepth model (dual-branch ViT + DPT decoder)
│   ├── dinov2.py               # DINOv2 Vision Transformer encoder
│   ├── flexible_attention.py   # Cross-attention w/ xFormers + SDPA fallback
│   │
│   ├── dinov2_layers/          # Vision Transformer building blocks (from Meta DINOv2)
│   │   ├── __init__.py
│   │   ├── attention.py        # Self-attention w/ optional xFormers (MemEffAttention)
│   │   ├── block.py            # Transformer encoder block (NestedTensorBlock)
│   │   ├── mlp.py              # Feed-forward network (Mlp)
│   │   ├── patch_embed.py      # Image → patch embeddings (PatchEmbed)
│   │   ├── swiglu_ffn.py       # SwiGLU activation FFN
│   │   ├── drop_path.py        # Stochastic depth regularization
│   │   └── layer_scale.py      # LayerScale normalization
│   │
│   └── util/                   # Utilities
│       ├── __init__.py
│       ├── blocks.py           # DPT decoder blocks (FeatureFusionBlock, ResidualConvUnit)
│       └── transform.py        # Image preprocessing (Resize, PrepareForNet)
│
├── tests/                      # Test suite (42 tests, runs in GitHub Actions)
│   ├── test_import.py          # Basic imports and smoke tests
│   └── test_model.py           # Architecture, forward pass, attention, preprocessing
│
├── example_data/               # Example RGB-D pairs for testing
│   ├── color_12.png            # RGB image sample
│   ├── depth_12.png            # Depth map sample
│   └── result.png              # Expected output
│
└── .github/workflows/          # CI/CD automation
    ├── test.yml                # Run tests on Python 3.10-3.12 (Ubuntu/macOS/Windows)
    ├── publish.yml             # Auto-publish to PyPI on release tags
    └── deploy-hf.yml           # Auto-deploy to HuggingFace Spaces on push to main
```

## Performance

### Accuracy

This implementation achieves **pixel-perfect alignment** with the ByteDance reference:
- ✅ **0 pixel difference** between vanilla and optimized inference (verified on test images)
- ✅ **Identical checkpoint loading** (weights are fully compatible)
- ✅ **Numerical precision preserved** (min=0.2036, max=1.1217, exact match)

CDMs achieve state-of-the-art performance on metric depth estimation:
- Superior accuracy compared to existing prompt-based depth models
- Zero-shot generalization across different camera types
- Real-time inference suitable for robot control (lightweight ViT variants)

**Performance optimizations:**
- xFormers support on CUDA (~8% faster than native SDPA)
- Mixed precision (FP16/BF16) for faster inference
- Device-specific optimizations (CUDA/MPS/CPU)

For detailed optimization strategies and benchmarks, see [OPTIMIZATION.md](OPTIMIZATION.md).

## What's Different from Reference?

This implementation maintains **100% compatibility** with ByteDance CDM while adding:

### 1. Performance Optimizations
- **xFormers support**: ~8% faster attention on CUDA (automatic fallback to SDPA)
- **torch.compile**: JIT compilation (CUDA only, auto-enabled)
- **Mixed precision**: FP16/BF16 support via `torch.amp.autocast`
- **Device-specific strategies**: Optimizations only where beneficial

### 2. Better CLI/API
- `--device` flag: Force specific device (auto/cuda/mps/cpu)
- `--precision` flag: Choose FP32/FP16/BF16
- `--no-optimize` flag: Disable optimizations for debugging
- Automatic device detection and optimization selection

### 3. Improved Architecture
- `FlexibleCrossAttention`: Inherits from `nn.MultiheadAttention` for checkpoint compatibility
- Automatic backend selection: xFormers (CUDA) → SDPA (fallback)
- Device-aware preprocessing: Uses model's device instead of auto-detection

### 4. Code Quality
- Type hints and better documentation
- Cleaner argument parsing
- Validation for precision/device combinations
- Helpful warnings for incompatible configurations

All changes are **backwards compatible** with original checkpoints and produce **identical numerical results**.

## Citation

If you use CDM in your research, please cite:

```bibtex
@article{liu2025manipulation,
  title={Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots},
  author={Liu, Minghuan and Zhu, Zhengbang and Han, Xiaoshen and Hu, Peng and Lin, Haotong and
          Li, Xinyao and Chen, Jingxiao and Xu, Jiafeng and Yang, Yichu and Lin, Yunfeng and
          Li, Xinghang and Yu, Yong and Zhang, Weinan and Kong, Tao and Kang, Bingyi},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

**Available on:** [GitHub](https://github.com/Aedelon/rgbd-depth) | [PyPI](https://pypi.org/project/rgbd-depth/) | [HF Spaces](https://huggingface.co/spaces/Aedelon/rgbd-depth)
