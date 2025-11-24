# Quickstart - Camera Depth Models

## 🚀 Démarrage en 30 secondes

```bash
# 1. Install
pip install camera-depth-models

# 2. Download un modèle
cdm-download --camera d435

# 3. Inférence
cdm-infer \
    --encoder vitl \
    --model-path <path-from-step-2> \
    --rgb-image example_data/color_12.png \
    --depth-image example_data/depth_12.png \
    --output result.png

# 4. Voir le résultat
open result.png  # macOS
# xdg-open result.png  # Linux
```

## 📚 Docs complètes

- **README** : [README.md](README.md)
- **Setup complet** : [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Optimisations** : [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)
- **Commandes** : [docs/CHEATSHEET.md](docs/CHEATSHEET.md)

## 🐍 Python API

```python
from rgbddepth import RGBDDepth, OptimizationConfig
import cv2, numpy as np, torch

# Auto-config
config = OptimizationConfig(device="auto")

# Load model
model = RGBDDepth(encoder='vitl', features=256,
                  out_channels=[256, 512, 1024, 1024], config=config)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Inference
rgb = cv2.imread('rgb.jpg')[:, :, ::-1]
depth = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED) / 1000.0
inv_depth = np.zeros_like(depth)
inv_depth[depth > 0] = 1 / depth[depth > 0]

pred = model.infer_image(rgb, inv_depth, input_size=518)
```

## 🆘 Aide

```bash
cdm-download --help
cdm-infer --help
```

## 🔗 Liens

- **GitHub** : https://github.com/TON-ORG/camera-depth-models
- **Modèles** : https://huggingface.co/collections/depth-anything/camera-depth-models-68b521181dedd223f4b020db
- **Papier** : https://manipulation-as-in-simulation.github.io/
