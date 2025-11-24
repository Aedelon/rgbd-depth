# Camera Depth Models (CDM)

CDM is a depth estimation package that uses Vision Transformer encoders for accurate RGB-D depth inference. It provides high-quality, metric depth data to enable seamless sim-to-real transfer for robotic manipulation tasks.

## Overview

Camera Depth Models (CDMs) are sensor-specific depth models trained to produce clean, simulation-like depth maps from noisy real-world depth camera data. By bridging the visual gap between simulation and reality through depth perception, CDMs enable robotic policies trained purely in simulation to transfer directly to real robots.

### Key Features

- **Metric Depth Estimation**: Produces accurate absolute depth measurements in meters
- **Multi-Camera Support**: Optimized models for various depth sensors (RealSense D405/D435/L515, ZED 2i, Azure Kinect)
- **Real-time Performance**: Lightweight inference for smooth robot control
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

## Installation

```bash
cd cdm
pip install -e .
```

## Usage

### Basic Inference

```bash
python infer.py \
    --encoder vitl \
    --model-path /path/to/model.pth \
    --rgb-image /path/to/rgb.jpg \
    --depth-image /path/to/depth.png \
    --output visualization.png
```

> We provide the example image and depth in the `cdm/input_data` folder. And the available models can be downloaded from [here](https://huggingface.co/collections/depth-anything/camera-depth-models-68b521181dedd223f4b020db).
### Command Line Arguments

- `--encoder`: Model size (`vits`, `vitb`, `vitl`, `vitg`) - default: `vitl`
- `--model-path`: Path to trained model checkpoint (required)
- `--rgb-image`: Path to RGB input image (required)
- `--depth-image`: Path to depth input image (required)
- `--output`: Output visualization path - default: `output.png`
- `--input-size`: Input resolution for inference - default: 518
- `--depth-scale`: Scale factor for depth values - default: 1000.0
- `--max-depth`: Maximum valid depth in meters - default: 6.0

### Python API

```python
from rgbddepth.dpt import RGBDDepth
import cv2
import numpy as np

# Load model
model = RGBDDepth(encoder='vitl', features=256)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load images
rgb = cv2.imread('rgb.jpg')[:, :, ::-1]  # BGR to RGB
depth = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED) / 1000.0  # Convert to meters

# Create similarity depth (inverse depth)
simi_depth = np.zeros_like(depth)
simi_depth[depth > 0] = 1 / depth[depth > 0]

# Run inference
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
cdm/
├── infer.py              # Main inference script
├── setup.py              # Package installation
├── rgbddepth/            # Core package
│   ├── __init__.py
│   ├── dpt.py            # Main RGBDDepth model
│   ├── dinov2.py         # DINOv2 encoder
│   ├── dinov2_layers/    # ViT transformer layers
│   └── util/             # Utility functions
│       ├── blocks.py     # Neural network blocks
│       └── transform.py  # Image preprocessing
└── README.md
```

## Performance

CDMs achieve state-of-the-art performance on metric depth estimation:
- Superior accuracy compared to existing prompt-based depth models
- Zero-shot generalization across different camera types
- Real-time inference suitable for robot control (lightweight ViT variants)

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

This project is licensed under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.
