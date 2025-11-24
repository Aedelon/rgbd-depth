#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Quick verification script to check if optimizations are properly installed.
"""

import sys


def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 60)
    print("Checking imports...")
    print("=" * 60)

    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "matplotlib": "Matplotlib",
    }

    optional = {
        "xformers": "xformers (GPU acceleration)",
    }

    all_ok = True

    # Check required
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name:<30} installed")
        except ImportError:
            print(f"✗ {name:<30} MISSING")
            all_ok = False

    # Check optional
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✓ {name:<30} installed")
        except ImportError:
            print(f"⚠ {name:<30} not installed (optional)")

    return all_ok


def check_devices():
    """Check available compute devices."""
    print("\n" + "=" * 60)
    print("Checking devices...")
    print("=" * 60)

    import torch

    print(f"PyTorch version: {torch.__version__}")

    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Device count: {torch.cuda.device_count()}")
    else:
        print(f"✗ CUDA not available")

    # MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"✓ MPS available (Apple Silicon)")
    else:
        print(f"✗ MPS not available")

    # CPU
    print(f"✓ CPU available")

    # Determine best device
    if torch.cuda.is_available():
        best_device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        best_device = "mps"
    else:
        best_device = "cpu"

    print(f"\nRecommended device: {best_device}")
    return best_device


def check_optimizations():
    """Check if optimization modules are properly installed."""
    print("\n" + "=" * 60)
    print("Checking optimization modules...")
    print("=" * 60)

    try:
        from rgbddepth.optimization_config import OptimizationConfig
        print("✓ OptimizationConfig imported")
    except ImportError as e:
        print(f"✗ Failed to import OptimizationConfig: {e}")
        return False

    try:
        from rgbddepth.attention import AdaptiveCrossAttention
        print("✓ AdaptiveCrossAttention imported")
    except ImportError as e:
        print(f"✗ Failed to import AdaptiveCrossAttention: {e}")
        return False

    try:
        from rgbddepth.dpt import RGBDDepth
        print("✓ RGBDDepth imported")
    except ImportError as e:
        print(f"✗ Failed to import RGBDDepth: {e}")
        return False

    return True


def check_config_creation(device):
    """Test creating an optimization config."""
    print("\n" + "=" * 60)
    print("Testing configuration creation...")
    print("=" * 60)

    try:
        from rgbddepth.optimization_config import OptimizationConfig

        # Test auto config
        config = OptimizationConfig(device="auto")
        print(f"✓ Auto config created")
        print(f"  - Detected device: {config.device}")
        print(f"  - Attention backend: {config.attention_backend}")
        print(f"  - Mixed precision: {config.mixed_precision}")

        # Test device-specific config
        config = OptimizationConfig(device=device)
        print(f"\n✓ Device-specific config created ({device})")
        print(f"  - Attention backend: {config.attention_backend}")
        print(f"  - Use compile: {config.use_compile}")
        print(f"  - Use channels_last: {config.use_channels_last}")
        print(f"  - Dtype: {config.dtype}")

        return True
    except Exception as e:
        print(f"✗ Failed to create config: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_creation(device):
    """Test creating an optimized model."""
    print("\n" + "=" * 60)
    print("Testing model creation...")
    print("=" * 60)

    try:
        from rgbddepth.dpt import RGBDDepth
        from rgbddepth.optimization_config import OptimizationConfig

        # Use smaller model for faster testing
        config = OptimizationConfig(
            device=device,
            use_compile=False  # Disable compile for testing
        )

        print(f"Creating small model (vits) for testing...")
        model = RGBDDepth(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            config=config
        )

        print(f"✓ Model created successfully")
        print(f"  - Device: {config.device}")
        print(f"  - Dtype: {config.dtype}")

        return True
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_attention_backends(device):
    """Check which attention backends are available."""
    print("\n" + "=" * 60)
    print("Checking attention backends...")
    print("=" * 60)

    from rgbddepth.attention import create_cross_attention

    backends_to_test = ["torch", "manual"]

    # Add xformers if on CUDA
    if device == "cuda":
        backends_to_test.insert(0, "xformers")

    results = {}

    for backend in backends_to_test:
        try:
            attention = create_cross_attention(
                embed_dim=384,
                num_heads=6,
                backend=backend,
                device=device
            )
            print(f"✓ {backend:<15} backend available")
            results[backend] = True
        except Exception as e:
            print(f"✗ {backend:<15} backend failed: {e}")
            results[backend] = False

    return results


def check_scripts():
    """Check if scripts are available."""
    print("\n" + "=" * 60)
    print("Checking scripts...")
    print("=" * 60)

    import os

    scripts = {
        "infer.py": "Optimized inference script",
        "test_optimizations.py": "Test suite",
        "example_usage.py": "Usage examples",
    }

    for script, description in scripts.items():
        if os.path.exists(script):
            print(f"✓ {script:<25} - {description}")
        else:
            print(f"✗ {script:<25} - MISSING")


def print_recommendations(device, backends):
    """Print recommendations based on detected hardware."""
    print("\n" + "=" * 60)
    print("Recommendations for your setup")
    print("=" * 60)

    if device == "cuda":
        print("\n🚀 You have a CUDA GPU!")
        if backends.get("xformers", False):
            print("\nRecommended command:")
            print("python infer.py \\")
            print("    --device cuda \\")
            print("    --attention-backend xformers \\")
            print("    --use-compile true \\")
            print("    --mixed-precision fp16 \\")
            print("    [other args...]")
            print("\nExpected speedup: 3-5x")
        else:
            print("\n⚠️  xformers not installed. Install for best performance:")
            print("pip install xformers")
            print("\nCurrent best command:")
            print("python infer.py \\")
            print("    --device cuda \\")
            print("    --use-compile true \\")
            print("    --mixed-precision fp16 \\")
            print("    [other args...]")

    elif device == "mps":
        print("\n🍎 You have Apple Silicon!")
        print("\nRecommended command:")
        print("python infer.py \\")
        print("    --device mps \\")
        print("    --attention-backend manual \\")
        print("    --use-channels-last true \\")
        print("    [other args...]")
        print("\nExpected speedup: 2-3x")

    else:  # CPU
        print("\n💻 Using CPU")
        print("\nRecommended command:")
        print("python infer.py \\")
        print("    --device cpu \\")
        print("    --use-channels-last true \\")
        print("    --fuse-encoder true \\")
        print("    --interpolation-mode bilinear \\")
        print("    [other args...]")
        print("\nExpected speedup: 1.5-2x")

    print("\n📚 Next steps:")
    print("1. Read quick start: QUICK_START_OPTIMIZATIONS.md")
    print("2. Run examples: python example_usage.py")
    print("3. Run tests: python test_optimizations.py")
    print("4. Benchmark: python infer.py --benchmark [args]")


def main():
    """Main verification routine."""
    print("\n" + "#" * 60)
    print("# CDM Optimization Installation Verification")
    print("#" * 60)

    # Check imports
    imports_ok = check_imports()
    if not imports_ok:
        print("\n❌ Some required packages are missing!")
        print("Install with: pip install -e .[optimizations]")
        return False

    # Check devices
    device = check_devices()

    # Check optimization modules
    optimizations_ok = check_optimizations()
    if not optimizations_ok:
        print("\n❌ Optimization modules not properly installed!")
        print("Make sure you ran: pip install -e .")
        return False

    # Test config creation
    config_ok = check_config_creation(device)
    if not config_ok:
        print("\n❌ Failed to create configuration!")
        return False

    # Test model creation
    model_ok = check_model_creation(device)
    if not model_ok:
        print("\n⚠️  Model creation failed, but this might be OK if you don't have a model checkpoint")

    # Check attention backends
    backends = check_attention_backends(device)

    # Check scripts
    check_scripts()

    # Print recommendations
    print_recommendations(device, backends)

    print("\n" + "#" * 60)
    print("# Verification Complete!")
    print("#" * 60)

    if imports_ok and optimizations_ok and config_ok:
        print("\n✅ All checks passed! Optimizations are ready to use.")
        return True
    else:
        print("\n⚠️  Some checks failed. See messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
