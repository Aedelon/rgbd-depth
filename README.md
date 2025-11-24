# Camera Depth Models (CDM)

[![PyPI version](https://badge.fury.io/py/camera-depth-models.svg)](https://badge.fury.io/py/camera-depth-models)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/manipulation-as-in-simulation/camera-depth-models/workflows/Tests/badge.svg)](https://github.com/Aedelon/camera-depth-models/actions)

Camera Depth Models (CDM) provides **accurate metric depth estimation** from RGB-D sensors using Vision Transformer encoders. CDM produces clean, simulation-like depth maps from noisy real-world camera data, enabling seamless **sim-to-real transfer** for robotic manipulation tasks.

## FAST Quick Start

### Installation

```bash
pip install camera-depth-models
```

Or install from source with GPU optimizations:

```bash
git clone https://github.com/manipulation-as-in-simulation/camera-depth-models.git
cd camera-depth-models
pip install -e .[cuda]  # For NVIDIA GPUs
# pip install -e .      # For CPU/MPS only
```

### Quickstart Demo

```bash
# Download the quickstart script
curl -O https://raw.githubusercontent.com/manipulation-as-in-simulation/camera-depth-models/main/scripts/quickstart.sh
chmod +x quickstart.sh
./quickstart.sh
```

### Basic Usage

```bash
# Download a pre-trained model
cdm-download --camera d435

# Run inference
cdm-infer \
    --encoder vitl \
    --model-path models/d435_model.pth \
    --rgb-image rgb.jpg \
    --depth-image depth.png \
    --output result.png
```

**That's it!** CDM automatically:
- Detects your hardware (CUDA/MPS/CPU)
- Applies optimal device-specific optimizations
- Produces high-quality metric depth output

## 📖 Features

- OK **Automatic optimization** - Zero configuration required
- OK **Metric depth estimation** - Accurate absolute depth in meters
- OK **Multi-camera support** - Pre-trained for RealSense, ZED 2i, Kinect
- OK **Sim-to-Real ready** - Clean depth from noisy sensor data
- OK **Real-time performance** - Optimized for robot control
- OK **Easy CLI** - Simple command-line tools
- OK **Python API** - Flexible programmatic interface

## 🎯 Supported Cameras

Pre-trained models available for:

| Camera | Models Available |
|--------|------------------|
| **Intel RealSense** | D405, D415, D435, D455, L515 |
| **Stereolabs ZED 2i** | Quality mode, Neural mode |
| **Microsoft Azure Kinect** | Standard mode |

Download with: `cdm-download --camera <model_name>`

List all models: `cdm-download --list`

## DESKTOP Python API

```python
from rgbddepth import RGBDDepth, OptimizationConfig
import cv2
import numpy as np

# Auto-optimized configuration
config = OptimizationConfig(device="auto")

# Create model
model = RGBDDepth(
    encoder='vitl',
    features=256,
    out_channels=[256, 512, 1024, 1024],
    config=config
)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load images
rgb = cv2.imread('rgb.jpg')[:, :, ::-1]  # BGR to RGB
depth = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED) / 1000.0  # to meters

# Create inverse depth
inv_depth = np.zeros_like(depth)
inv_depth[depth > 0] = 1 / depth[depth > 0]

# Run inference
pred_depth = model.infer_image(rgb, inv_depth, input_size=518)
```

## 📊 Performance

**Device-Specific Optimizations** deliver **1.8x to 4.8x speedup**:

| Device | Speed | Optimizations |
|--------|-------|---------------|
| **NVIDIA GPU** | **4-5x faster** | xformers + compile + FP16 |
| **Apple Silicon** | **2-3x faster** | Manual attention + fused encoder |
| **CPU** | **1.5-2x faster** | channels_last + fused encoder |

See [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) for details.

## 🔧 Advanced Usage

### Manual Configuration

```python
from rgbddepth import OptimizationConfig

# Fine-grained control
config = OptimizationConfig(
    device="cuda",
    attention_backend="xformers",
    use_compile=True,
    mixed_precision="fp16",
    use_channels_last=True
)
```

### Command-Line Options

```bash
cdm-infer \
    --encoder vitl \
    --model-path model.pth \
    --rgb-image rgb.jpg \
    --depth-image depth.png \
    --output result.png \
    --device cuda \
    --attention-backend xformers \
    --mixed-precision fp16 \
    --use-compile true
```

See `cdm-infer --help` for all options.

## 🏗️ Architecture

CDM uses a dual-branch Vision Transformer architecture:

- **RGB Branch**: Extracts semantic information from RGB images
- **Depth Branch**: Processes noisy depth sensor data
- **Cross-Attention Fusion**: Combines RGB semantics with depth scale
- **DPT Decoder**: Produces final metric depth estimation

Supported encoder sizes: `vits`, `vitb`, `vitl`, `vitg`

## DOCS Documentation

- [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) - Complete optimization guide
- [CHEATSHEET.md](docs/CHEATSHEET.md) - Quick reference commands

## 🔬 Research

CDM is part of the **"Manipulation as in Simulation"** research project. For more details, see:

- **Paper**: [Manipulation as in Simulation](https://manipulation-as-in-simulation.github.io/)
- **Full Suite**: [manip-as-in-sim-suite](https://github.com/manipulation-as-in-simulation/manip-as-in-sim-suite) (includes WBCMimic for robot demonstrations)
- **Dataset**: [ByteCameraDepth](https://huggingface.co/datasets/ByteDance-Seed/ByteCameraDepth)

### Citation

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

## 🧪 Testing

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_import.py -v
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `pytest tests/`
5. Format code: `black . && isort .`
6. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## 🔗 Links

- **GitHub**: [camera-depth-models](https://github.com/manipulation-as-in-simulation/camera-depth-models)
- **PyPI**: [camera-depth-models](https://pypi.org/project/camera-depth-models/)
- **Pre-trained Models**: [HuggingFace Collection](https://huggingface.co/collections/depth-anything/camera-depth-models-68b521181dedd223f4b020db)
- **Project Website**: [Manipulation as in Simulation](https://manipulation-as-in-simulation.github.io/)

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/manipulation-as-in-simulation/camera-depth-models/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manipulation-as-in-simulation/camera-depth-models/discussions)

---

Open-source contribution based on the [Manipulation as in Simulation](https://manipulation-as-in-simulation.github.io/) research project.
