#!/bin/bash
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=========================================="
echo "Camera Depth Models - Quickstart Demo"
echo "=========================================="
echo ""

# Check if running in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the camera-depth-models root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

# Install package if not already installed
if ! python3 -c "import rgbddepth" 2>/dev/null; then
    echo ""
    echo "Installing camera-depth-models package..."
    pip install -e . -q
    echo "✓ Package installed"
fi

# Create models directory
mkdir -p models

# Download example model (D435)
echo ""
echo "Downloading example model (RealSense D435)..."
if [ ! -f "models/d435_model.pth" ]; then
    echo "Note: This will download from HuggingFace (~400MB)"
    echo "Install huggingface-hub if needed: pip install huggingface-hub"

    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil

print('Downloading model...')
path = hf_hub_download(
    repo_id='depth-anything/camera-depth-model-d435',
    filename='model.pth',
    cache_dir='./models'
)
# Copy to predictable location
shutil.copy(path, './models/d435_model.pth')
print('✓ Model downloaded')
"
    else
        echo "⚠ huggingface-hub not installed. Skipping download."
        echo "  Install with: pip install huggingface-hub"
        echo "  Or use: cdm-download --camera d435"
        echo ""
        echo "For this demo, using a placeholder..."
        touch models/d435_model.pth  # Placeholder
    fi
else
    echo "✓ Model already downloaded"
fi

# Check example data exists
if [ ! -d "example_data" ]; then
    echo ""
    echo "Error: example_data directory not found"
    echo "Please ensure example_data/ is present with color_12.png and depth_12.png"
    exit 1
fi

# Run inference
echo ""
echo "Running inference on example data..."
echo "Input: example_data/color_12.png + example_data/depth_12.png"
echo ""

if [ -f "models/d435_model.pth" ] && [ -s "models/d435_model.pth" ]; then
    python3 -m rgbddepth.infer \
        --encoder vitl \
        --model-path models/d435_model.pth \
        --rgb-image example_data/color_12.png \
        --depth-image example_data/depth_12.png \
        --output quickstart_result.png

    echo ""
    echo "=========================================="
    echo "✓ Quickstart complete!"
    echo "=========================================="
    echo ""
    echo "Output saved to: quickstart_result.png"
    echo ""
    echo "Next steps:"
    echo "  - View the result: open quickstart_result.png"
    echo "  - Download more models: cdm-download --list"
    echo "  - See optimizations: cat docs/OPTIMIZATIONS.md"
    echo "  - Run on your data: cdm-infer --help"
else
    echo ""
    echo "Model download was skipped. To complete quickstart:"
    echo "  1. Install huggingface-hub: pip install huggingface-hub"
    echo "  2. Download model: cdm-download --camera d435"
    echo "  3. Run inference: cdm-infer --encoder vitl --model-path <path> --rgb-image example_data/color_12.png --depth-image example_data/depth_12.png"
fi
