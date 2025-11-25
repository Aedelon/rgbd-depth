# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2025-01-25

### Added
- **xFormers support** for ~8% faster CUDA inference with automatic fallback to SDPA
- **Mixed precision support** (FP16/BF16) via `--precision` flag
- **Device selection** via `--device` flag (auto/cuda/mps/cpu)
- **FlexibleCrossAttention** module for optimized cross-attention with checkpoint compatibility
- **Comprehensive documentation**: README.md with feature comparison, OPTIMIZATION.md guide
- **CI/CD workflows**: Automated tests on push/PR, PyPI publication on release
- **Entry point**: `rgbd-depth` CLI command for easy usage

### Changed
- **Package renamed** from `camera-depth-models` to `rgbd-depth` for PyPI
- **Optimizations auto-enabled** on CUDA by default (use `--no-optimize` to disable)
- **Simplified API**: Removed `OptimizationConfig`, streamlined to `RGBDDepth` with `use_xformers` flag
- **Updated dependencies**: Requires PyTorch 2.0+ for SDPA support
- **Cleaned up codebase**: Removed obsolete scripts, docs, and tests

### Fixed
- **Precision preservation**: Pixel-perfect alignment with ByteDance reference (0 pixel diff)
- **Device detection**: Model preprocessing now uses correct device
- **MPS compatibility**: Proper fallback when xFormers/torch.compile not beneficial

### Performance
- **CUDA**: ~8% faster with xFormers (FP32), ~2× faster with FP16/BF16
- **MPS**: 1.34s baseline, torch.compile provides no gain (disabled by default)
- **CPU**: 13.37s baseline, torch.compile counterproductive (disabled by default)

## [1.0.1] - 2024-XX-XX

### Added
- Initial fork from ByteDance Camera Depth Models
- Basic optimization support

## [1.0.0] - 2024-XX-XX

### Added
- Original ByteDance CDM implementation
- Support for RealSense D405/D435/L515, ZED 2i, Azure Kinect
- DINOv2 Vision Transformer encoder
- DPT decoder with cross-attention fusion

---

**Note:** This package maintains 100% compatibility with original ByteDance checkpoints.
All weights are interchangeable between this optimized version and the reference implementation.
