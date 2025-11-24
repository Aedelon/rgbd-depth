# CDM Optimization Cheatsheet

Quick reference for using optimized CDM inference.

## 🚀 Quick Commands

### Auto-optimized (recommended)
```bash
python infer.py \
    --encoder vitl \
    --model-path model.pth \
    --rgb-image rgb.jpg \
    --depth-image depth.png
```

### CUDA - Maximum Speed
```bash
python infer.py \
    --encoder vitl --model-path model.pth \
    --rgb-image rgb.jpg --depth-image depth.png \
    --device cuda --attention-backend xformers \
    --use-compile true --mixed-precision fp16
```

### MPS - Apple Silicon
```bash
python infer.py \
    --encoder vitl --model-path model.pth \
    --rgb-image rgb.jpg --depth-image depth.png \
    --device mps --attention-backend manual \
    --use-channels-last true
```

### CPU - Best Performance
```bash
python infer.py \
    --encoder vitl --model-path model.pth \
    --rgb-image rgb.jpg --depth-image depth.png \
    --device cpu --use-channels-last true \
    --fuse-encoder true --interpolation-mode bilinear
```

### Benchmark
```bash
python infer.py \
    --encoder vitl --model-path model.pth \
    --rgb-image rgb.jpg --depth-image depth.png \
    --benchmark --benchmark-iters 20
```

## 📊 Compare Backends

```bash
# Compare all backends on your device
for backend in torch manual xformers; do
    echo "=== Testing $backend ==="
    python infer.py \
        --attention-backend $backend \
        --benchmark --benchmark-iters 10 \
        --encoder vitl --model-path model.pth \
        --rgb-image rgb.jpg --depth-image depth.png
done
```

## 🔧 Common Configurations

### Quality Mode (slower but best quality)
```bash
python infer.py \
    --mixed-precision fp32 \
    --interpolation-mode bicubic \
    [other args...]
```

### Speed Mode (fastest)
```bash
python infer.py \
    --mixed-precision fp16 \
    --use-compile true \
    --interpolation-mode nearest \
    [other args...]
```

### Memory-Efficient Mode
```bash
python infer.py \
    --mixed-precision fp16 \
    --fuse-encoder true \
    --input-size 448 \
    [other args...]
```

### Balanced Mode
```bash
python infer.py \
    --mixed-precision auto \
    --interpolation-mode bilinear \
    [other args...]
```

## 🐍 Python API

### Basic Usage
```python
from rgbddepth.dpt import RGBDDepth
from rgbddepth.optimization_config import OptimizationConfig

config = OptimizationConfig(device="auto")
model = RGBDDepth(encoder='vitl', config=config)
pred = model.infer_image(rgb, depth, input_size=518)
```

### CUDA Configuration
```python
config = OptimizationConfig(
    device="cuda",
    attention_backend="xformers",
    use_compile=True,
    mixed_precision="fp16"
)
model = RGBDDepth(encoder='vitl', config=config)
```

### MPS Configuration
```python
config = OptimizationConfig(
    device="mps",
    attention_backend="manual",
    use_channels_last=True,
    mixed_precision="fp32"
)
model = RGBDDepth(encoder='vitl', config=config)
```

### CPU Configuration
```python
config = OptimizationConfig(
    device="cpu",
    use_channels_last=True,
    fuse_depth_encoder=True,
    interpolation_mode="bilinear"
)
model = RGBDDepth(encoder='vitl', config=config)
```

## 🧪 Testing

```bash
# Run full test suite
python test_optimizations.py

# Run examples
python example_usage.py

# Test specific configuration
python infer.py --benchmark \
    --device cuda --attention-backend xformers \
    [other args...]
```

## 🔍 Debugging

### Enable verbose output
```python
config = OptimizationConfig(device="auto")
print(config.summary())  # Print configuration
```

### Check available backends
```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available())
try:
    import xformers
    print('xformers: available')
except:
    print('xformers: not available')
"
```

### Validate installation
```bash
pip list | grep -E "torch|xformers|opencv"
```

## 📦 Installation Commands

```bash
# Basic
pip install -e .

# With optimizations
pip install -e .[optimizations]

# Development
pip install -e .[dev]

# Everything
pip install -e .[all]
```

## ⚠️ Troubleshooting

### xformers not found
```bash
pip install xformers
# Or use: --attention-backend torch
```

### Out of memory
```bash
# Try in order:
--mixed-precision fp16
--fuse-encoder true
--input-size 448
--encoder vitb
```

### MPS crashes
```bash
--device mps \
--attention-backend manual \
--mixed-precision fp32
```

### Slow CPU inference
```bash
--use-channels-last true \
--fuse-encoder true \
--interpolation-mode bilinear
```

## 📈 Optimization Flags Reference

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--device` | auto/cuda/mps/cpu | auto | Target device |
| `--attention-backend` | auto/xformers/torch/manual | auto | Attention impl |
| `--use-compile` | auto/true/false | auto | torch.compile |
| `--mixed-precision` | auto/fp16/bf16/fp32 | auto | Precision |
| `--use-channels-last` | auto/true/false | auto | Memory layout |
| `--fuse-encoder` | auto/true/false | auto | Share encoders |
| `--interpolation-mode` | bicubic/bilinear/nearest | bilinear | Upsampling |

## 🎯 Performance Tips

### For Speed
1. Use `--attention-backend xformers` (CUDA)
2. Enable `--use-compile true` (CUDA)
3. Use `--mixed-precision fp16` (GPU)
4. Choose `--interpolation-mode nearest`

### For Quality
1. Use `--mixed-precision fp32`
2. Use `--interpolation-mode bicubic`
3. Use larger `--input-size`
4. Use larger encoder (vitg > vitl > vitb)

### For Memory
1. Use `--mixed-precision fp16`
2. Enable `--fuse-encoder true`
3. Reduce `--input-size`
4. Use smaller encoder (vits < vitb < vitl)

## 📚 Documentation Links

- **Quick Start**: `QUICK_START_OPTIMIZATIONS.md`
- **Full Docs**: `OPTIMIZATIONS.md`
- **Summary**: `OPTIMIZATION_SUMMARY.md`
- **Examples**: `example_usage.py`
- **Tests**: `test_optimizations.py`

## 🚀 One-Liners

### Quick test
```bash
python infer.py --encoder vitl --model-path model.pth --rgb-image rgb.jpg --depth-image depth.png
```

### Benchmark all configs
```bash
for config in "cuda+xformers" "cuda+torch" "mps+manual" "cpu+torch"; do
    IFS='+' read -r device backend <<< "$config"
    echo "Testing $config"
    python infer.py --device $device --attention-backend $backend --benchmark [args]
done
```

### Compare precision modes
```bash
for prec in fp32 fp16 bf16; do
    echo "Testing $prec"
    python infer.py --mixed-precision $prec --benchmark [args]
done
```

### Memory usage test
```bash
python -c "
import torch
from rgbddepth.dpt import RGBDDepth
from rgbddepth.optimization_config import OptimizationConfig

config = OptimizationConfig(device='cuda', mixed_precision='fp16')
model = RGBDDepth(encoder='vitl', config=config)

print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')
"
```

## 💾 Save/Load Optimized Model

```python
# Save
torch.save(model.state_dict(), 'optimized_model.pth')

# Load
model = RGBDDepth(encoder='vitl', config=config)
model.load_state_dict(torch.load('optimized_model.pth'))
model.eval()
```

## 🔄 Batch Processing

```python
import glob
from pathlib import Path

rgb_images = sorted(glob.glob('data/rgb/*.jpg'))
depth_images = sorted(glob.glob('data/depth/*.png'))

for rgb_path, depth_path in zip(rgb_images, depth_images):
    rgb = load_rgb(rgb_path)
    depth = load_depth(depth_path)
    pred = model.infer_image(rgb, depth)
    save_prediction(pred, output_path)
```

## 📊 Performance Matrix

| Device | Backend | Compile | Precision | Speed | Memory |
|--------|---------|---------|-----------|-------|--------|
| CUDA | xformers | ✓ | fp16 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CUDA | torch | ✓ | fp16 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| MPS | manual | ✗ | fp32 | ⭐⭐⭐ | ⭐⭐⭐ |
| CPU | torch | ✗ | fp32 | ⭐⭐ | ⭐⭐ |

---

**Tip**: Use `--benchmark` to find the best configuration for your hardware!
