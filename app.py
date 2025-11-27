#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Gradio demo for rgbd-depth on Hugging Face Spaces."""

import logging
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from rgbddepth import RGBDDepth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global model cache
MODELS = {}

# Model mappings from HuggingFace (all are vitl encoder)
# Format: "camera_model": ("repo_id", "checkpoint_filename")
HF_MODELS = {
    "d435": ("depth-anything/camera-depth-model-d435", "cdm_d435.ckpt"),
    "d405": ("depth-anything/camera-depth-model-d405", "cdm_d405.ckpt"),
    "l515": ("depth-anything/camera-depth-model-l515", "cdm_l515.ckpt"),
    "zed2i": ("depth-anything/camera-depth-model-zed2i", "cdm_zed2i.ckpt"),
}

# Default model
DEFAULT_MODEL = "d435"


def download_model(camera_model: str = DEFAULT_MODEL):
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download

        repo_id, filename = HF_MODELS.get(camera_model, HF_MODELS[DEFAULT_MODEL])
        logger.info(f"Downloading {camera_model} model from {repo_id}/{filename}...")

        # Download the checkpoint
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=".cache")

        logger.info(f"Downloaded to {checkpoint_path}")
        return checkpoint_path

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


def load_model(camera_model: str = DEFAULT_MODEL, use_xformers: bool = False):
    """Load model with automatic download from HuggingFace."""
    cache_key = f"{camera_model}_{use_xformers}"

    if cache_key not in MODELS:
        # All HF models use vitl encoder
        config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
            "use_xformers": use_xformers,
        }

        model = RGBDDepth(**config)

        # Try to load weights
        checkpoint_path = None

        # 1. Try local checkpoints/ directory first
        local_path = Path(f"checkpoints/{camera_model}.pt")
        if local_path.exists():
            checkpoint_path = str(local_path)
            logger.info(f"Using local checkpoint: {checkpoint_path}")
        else:
            # 2. Download from HuggingFace
            checkpoint_path = download_model(camera_model)

        # Load checkpoint if available
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "model" in checkpoint:
                    states = {k[7:]: v for k, v in checkpoint["model"].items()}
                elif "state_dict" in checkpoint:
                    states = {k[9:]: v for k, v in checkpoint["state_dict"].items()}
                else:
                    states = checkpoint

                model.load_state_dict(states, strict=False)
                logger.info(f"Loaded checkpoint for {camera_model}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, using random weights")
        else:
            logger.warning(
                f"No checkpoint available for {camera_model}, using random weights (demo only)"
            )

        # Move to GPU if available (CUDA or MPS for macOS)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        model = model.to(device).eval()

        MODELS[cache_key] = model

    return MODELS[cache_key]


def process_depth(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_model: str = DEFAULT_MODEL,
    input_size: int = 518,
    depth_scale: float = 1000.0,
    max_depth: float = 25.0,
    use_xformers: bool = False,
    precision: str = "fp32",
    colormap: str = "Spectral",
) -> tuple[Image.Image, str]:
    """Process RGB-D depth refinement.

    Args:
        rgb_image: RGB image as numpy array [H, W, 3]
        depth_image: Depth image as numpy array [H, W] or [H, W, 3]
        camera_model: Camera model to use (d435, d405, l515, zed2i)
        input_size: Input size for inference
        depth_scale: Scale factor for depth values
        max_depth: Maximum valid depth value
        use_xformers: Whether to use xFormers (CUDA only)
        precision: Precision mode (fp32/fp16/bf16)
        colormap: Matplotlib colormap for visualization

    Returns:
        Tuple of (refined depth image, info message)
    """
    try:
        # Validate inputs
        if rgb_image is None:
            return None, "❌ Please upload an RGB image"
        if depth_image is None:
            return None, "❌ Please upload a depth image"

        # Convert depth to single channel if needed
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]

        # Normalize depth
        depth_normalized = depth_image.astype(np.float32) / depth_scale
        depth_normalized[depth_normalized > max_depth] = 0.0

        # Create inverse depth (similarity depth)
        simi_depth = np.zeros_like(depth_normalized)
        valid_mask = depth_normalized > 0
        simi_depth[valid_mask] = 1.0 / depth_normalized[valid_mask]

        # Load model
        model = load_model(camera_model, use_xformers and torch.cuda.is_available())
        device = next(model.parameters()).device

        # Determine precision
        if precision == "fp16" and device.type in ["cuda", "mps"]:
            dtype = torch.float16
        elif precision == "bf16" and device.type == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = None  # FP32

        # Log input statistics
        logger.debug(f"depth_image raw: min={depth_image.min():.1f}, max={depth_image.max():.1f}")
        logger.debug(
            f"depth_normalized: min={depth_normalized[depth_normalized>0].min():.4f}, max={depth_normalized.max():.4f}"
        )
        logger.debug(
            f"simi_depth: min={simi_depth[simi_depth>0].min():.4f}, max={simi_depth.max():.4f}"
        )

        # Run inference
        if dtype is not None:
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                pred = model.infer_image(rgb_image, simi_depth, input_size=input_size)
        else:
            pred = model.infer_image(rgb_image, simi_depth, input_size=input_size)

        # Log prediction statistics
        logger.debug(f"pred (inverse depth): min={pred[pred>0].min():.4f}, max={pred.max():.4f}")

        # Convert from inverse depth to depth
        pred = np.where(pred > 1e-8, 1.0 / pred, 0.0)

        # Log final depth statistics
        logger.debug(f"pred (depth): min={pred[pred>0].min():.4f}, max={pred.max():.4f}")

        # Colorize for visualization
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Normalize to [0, 1]
            pred_min, pred_max = pred.min(), pred.max()
            if pred_max - pred_min > 1e-8:
                pred_norm = (pred - pred_min) / (pred_max - pred_min)
            else:
                pred_norm = np.zeros_like(pred)

            # Apply colormap
            cm_func = matplotlib.colormaps[colormap]
            pred_colored = cm_func(pred_norm, bytes=True)[:, :, :3]  # RGB only

            # Create PIL Image
            output_image = Image.fromarray(pred_colored)

        except ImportError:
            # Fallback to grayscale if matplotlib not available
            pred_norm = ((pred - pred.min()) / (pred.max() - pred.min() + 1e-8) * 255).astype(
                np.uint8
            )
            output_image = Image.fromarray(pred_norm, mode="L").convert("RGB")

        # Create info message
        info = f"""
✅ **Refinement complete!**

**Camera Model:** {camera_model.upper()}
**Precision:** {precision.upper()}
**Device:** {device.type.upper()}
**Input size:** {input_size}px
**Depth range:** {pred_min:.3f}m - {pred_max:.3f}m
**xFormers:** {'✓ Enabled' if use_xformers and torch.cuda.is_available() else '✗ Disabled'}
"""

        return output_image, info.strip()

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="rgbd-depth Demo") as demo:
    gr.Markdown(
        """
    # 🎨 rgbd-depth: RGB-D Depth Refinement

    High-quality depth map refinement using Vision Transformers. Based on [ByteDance's camera-depth-models](https://manipulation-as-in-simulation.github.io/).

    📥 **Models are automatically downloaded from Hugging Face on first use!**

    Choose your camera model (D435, D405, L515, or ZED 2i) and the trained weights will be downloaded automatically.
    """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📥 Inputs")

            rgb_input = gr.Image(
                label="RGB Image",
                type="numpy",
                height=300,
            )

            depth_input = gr.Image(
                label="Input Depth Map",
                type="numpy",
                height=300,
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                camera_choice = gr.Dropdown(
                    choices=["d435", "d405", "l515", "zed2i"],
                    value=DEFAULT_MODEL,
                    label="Camera Model",
                    info="Choose the camera model for trained weights (auto-downloads from HF)",
                )

                input_size = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=518,
                    step=2,
                    label="Input Size",
                    info="Resolution for processing (higher = better but slower)",
                )

                depth_scale = gr.Number(
                    value=1000.0,
                    label="Depth Scale",
                    info="Scale factor to convert depth values to meters",
                )

                max_depth = gr.Number(
                    value=25.0,
                    label="Max Depth (m)",
                    info="Maximum valid depth value",
                )

                precision_choice = gr.Radio(
                    choices=["fp32", "fp16", "bf16"],
                    value="fp32",
                    label="Precision",
                    info="fp16/bf16 = faster but slightly less accurate (CUDA only)",
                )

                use_xformers = gr.Checkbox(
                    value=False,  # Set to True to test xFormers by default
                    label="Use xFormers (CUDA only)",
                    info="~8% faster on CUDA with xFormers installed",
                )

                colormap_choice = gr.Dropdown(
                    choices=["Spectral", "viridis", "plasma", "inferno", "magma", "turbo"],
                    value="Spectral",
                    label="Colormap",
                    info="Visualization colormap",
                )

            process_btn = gr.Button("🚀 Refine Depth", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### 📤 Output")

            output_image = gr.Image(
                label="Refined Depth Map",
                type="pil",
                height=600,
            )

            output_info = gr.Markdown()

    # Example inputs
    gr.Markdown("### 📸 Examples")
    gr.Examples(
        examples=[
            ["example_data/color_12.png", "example_data/depth_12.png"],
        ],
        inputs=[rgb_input, depth_input],
        label="Try with example images",
    )

    # Process button click
    process_btn.click(
        fn=process_depth,
        inputs=[
            rgb_input,
            depth_input,
            camera_choice,
            input_size,
            depth_scale,
            max_depth,
            use_xformers,
            precision_choice,
            colormap_choice,
        ],
        outputs=[output_image, output_info],
    )

    # Footer
    gr.Markdown(
        """
    ---

    ### 🔗 Links

    - **GitHub:** [Aedelon/camera-depth-models](https://github.com/Aedelon/camera-depth-models)
    - **PyPI:** [rgbd-depth](https://pypi.org/project/rgbd-depth/)
    - **Paper:** [Manipulation-as-in-Simulation](https://manipulation-as-in-simulation.github.io/)

    ### 📦 Install

    ```bash
    pip install rgbd-depth
    ```

    ### 💻 CLI Usage

    ```bash
    rgbd-depth \\
      --model-path model.pt \\
      --rgb-image input.jpg \\
      --depth-image depth.png \\
      --output refined.png
    ```

    ---

    Built with ❤️ by [Aedelon](https://github.com/Aedelon) | Powered by [Gradio](https://gradio.app)
    """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
