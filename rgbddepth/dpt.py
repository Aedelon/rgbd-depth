#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Unified RGBDDepth model with built-in device-specific optimizations.
This merges the previous optimized implementation into the main class.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from typing import Optional

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import NormalizeImage, PrepareForNet, Resize
from .attention import create_cross_attention
from .optimization_config import OptimizationConfig


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        sigact_out=False,
        interpolation_mode="bilinear",
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.interpolation_mode = interpolation_mode

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )

        if not sigact_out:
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
        else:
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode=self.interpolation_mode,
            align_corners=True if self.interpolation_mode != "nearest" else None,
        )
        out = self.scratch.output_conv2(out)

        return out


class RGBDDepth(nn.Module):
    """RGBDDepth model with built-in device-specific optimizations."""

    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0,
        config: Optional[OptimizationConfig] = None,
    ):
        super(RGBDDepth, self).__init__()

        # Use provided config or create default
        self.config = config if config is not None else OptimizationConfig()

        print(self.config.summary())

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }

        self.max_depth = max_depth
        self.encoder = encoder

        # Initialize encoders
        if self.config.fuse_depth_encoder:
            print("Using fused encoder (memory efficient)")
            self.pretrained = DINOv2(model_name=encoder)
            self.depth_pretrained = self.pretrained
        else:
            print("Using separate encoders (better parallelism)")
            self.pretrained = DINOv2(model_name=encoder)
            self.depth_pretrained = DINOv2(model_name=encoder)

        # Initialize depth head with optimized interpolation
        self.depth_head_rgbd = DPTHead(
            self.pretrained.embed_dim * 2,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            sigact_out=False,
            interpolation_mode=self.config.interpolation_mode,
        )

        # Create optimized cross-attention modules
        num_heads = 4
        self.crossAtts = nn.ModuleList(
            [
                create_cross_attention(
                    embed_dim=self.pretrained.embed_dim,
                    num_heads=num_heads,
                    backend=self.config.attention_backend,
                    device=self.config.device,
                )
                for _ in range(4)
            ]
        )

        # Apply device-specific optimizations
        self._apply_optimizations()

    def _apply_optimizations(self):
        """Apply device-specific optimizations."""
        self.to(self.config.device)

        if self.config.use_channels_last:
            print("Applying channels_last memory format to decoder")
            self.depth_head_rgbd.to(memory_format=torch.channels_last)

        if self.config.dtype != torch.float32:
            print(f"Converting model to {self.config.dtype}")
            try:
                self.to(dtype=self.config.dtype)
            except Exception as e:
                if self.config.device.startswith("mps") and self.config.dtype == torch.float16:
                    print(f"Warning: FP16 failed on MPS ({e}), falling back to FP32")
                    self.config.dtype = torch.float32
                    self.to(dtype=torch.float32)
                else:
                    raise

        if self.config.use_compile:
            try:
                print("Applying torch.compile optimization")
                self.forward = torch.compile(
                    self.forward,
                    mode="max-autotune",
                    fullgraph=False,
                )
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

    def forward(self, x):
        rgb, depth = x[:, :3], x[:, 3:]
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        with torch.no_grad():
            features_rgb = self.pretrained.get_intermediate_layers(
                rgb, self.intermediate_layer_idx[self.encoder], return_class_token=True
            )

        depth_input = depth.repeat(1, 3, 1, 1)

        features_depth = self.depth_pretrained.get_intermediate_layers(
            depth_input,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        features = []
        for f_rgb, f_depth, crossAtt in zip(
            features_rgb, features_depth, self.crossAtts
        ):
            B, N, C = f_rgb[0].shape

            token_feat = torch.stack([f_rgb[0], f_depth[0]], dim=2)
            token_feat = token_feat.reshape(B * N, 2, C)

            att_feat, _ = crossAtt(token_feat, token_feat, token_feat)
            att_feat = att_feat.sum(dim=1).reshape(B, N, C)

            feat = torch.cat([f_rgb[0], att_feat], dim=2)
            cls_t = torch.cat([f_rgb[1], f_depth[1]], dim=1)

            features.append((feat, cls_t))

        depth = self.depth_head_rgbd(features, patch_h, patch_w)
        depth = F.relu(depth)

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, depth_low_res, input_size=518):
        """Run inference on a single image."""
        inputs, (h, w) = self.image2tensor(raw_image, depth_low_res, input_size)

        device_type = self.config.device.split(":")[0]
        if self.config.dtype != torch.float32 and device_type in ["cuda", "mps"]:
            try:
                with torch.autocast(device_type=device_type, dtype=self.config.dtype):
                    pred_depth = self.forward(inputs)
            except Exception as e:
                print(f"Warning: autocast failed ({e}), using FP32")
                pred_depth = self.forward(inputs)
        else:
            pred_depth = self.forward(inputs)

        pred_depth = F.interpolate(pred_depth[:, None], (h, w), mode="nearest")[0, 0]
        return pred_depth.cpu().numpy()

    def image2tensor(self, raw_image, depth, input_size=518):
        """Convert image and depth to tensor format."""
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = raw_image.shape[:2]

        if raw_image.shape[2] == 3 and raw_image[:, :, 0].mean() != raw_image[:, :, 2].mean():
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        else:
            image = raw_image.astype(float) / 255.0

        prepared = transform({"image": image, "depth": depth})
        image = prepared["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        depth = prepared["depth"]
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

        inputs = torch.cat((image, depth), dim=1)
        inputs = inputs.to(device=self.config.device, dtype=self.config.dtype)

        return inputs, (h, w)

    def get_config(self) -> OptimizationConfig:
        return self.config
