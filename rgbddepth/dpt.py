#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import NormalizeImage, PrepareForNet, Resize


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


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        sigact_out=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

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
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        return out


class RGBDDepth(nn.Module):
    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0,
    ):
        super(RGBDDepth, self).__init__()

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }

        self.max_depth = max_depth

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_pretrained = DINOv2(model_name=encoder)

        # self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, sigact_out=False)
        self.depth_head_rgbd = DPTHead(
            self.pretrained.embed_dim * 2,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            sigact_out=False,
        )

        # cross att
        num_heads = 4
        self.crossAtts = nn.ModuleList(
            [
                nn.MultiheadAttention(self.pretrained.embed_dim, num_heads, batch_first=True),
                nn.MultiheadAttention(self.pretrained.embed_dim, num_heads, batch_first=True),
                nn.MultiheadAttention(self.pretrained.embed_dim, num_heads, batch_first=True),
                nn.MultiheadAttention(self.pretrained.embed_dim, num_heads, batch_first=True),
            ]
        )

    def forward(self, x):
        rgb, depth = x[:, :3], x[:, 3:]
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        with torch.no_grad():
            features_rgb = self.pretrained.get_intermediate_layers(
                rgb, self.intermediate_layer_idx[self.encoder], return_class_token=True
            )

        features_depth = self.depth_pretrained.get_intermediate_layers(
            depth.repeat(1, 3, 1, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        features = []
        for f_rgb, f_depth, crossAtt in zip(features_rgb, features_depth, self.crossAtts):
            B, N, C = f_rgb[0].shape
            tf_rgb = f_rgb[0].reshape(B * N, 1, C)
            tf_depth = f_depth[0].reshape(B * N, 1, C)
            token_feat = torch.concat((tf_rgb, tf_depth), axis=1)
            att_feat, _ = crossAtt(token_feat, token_feat, token_feat)
            att_feat = att_feat.reshape(B * N, 2, C).sum(axis=1).reshape(B, N, C)

            feat = torch.concat((f_rgb[0], att_feat), axis=2)
            cls_t = torch.concat((f_rgb[1], f_depth[1]), axis=1)
            tuples = (feat, cls_t)
            features.append(tuples)
        depth = self.depth_head_rgbd(features, patch_h, patch_w)
        depth = F.relu(depth)
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, depth_low_res, input_size=518):
        inputs, (h, w) = self.image2tensor(raw_image, depth_low_res, input_size)
        pred_depth = self.forward(inputs)
        pred_depth = F.interpolate(pred_depth[:, None], (h, w), mode="nearest")[0, 0]
        return pred_depth.cpu().numpy()

    def image2tensor(self, raw_image, depth, input_size=518):
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

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        prepared = transform({"image": image, "depth": depth})
        image = prepared["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        depth = prepared["depth"]
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

        inputs = torch.cat((image, depth), dim=1)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        inputs = inputs.to(DEVICE)

        return inputs, (h, w)
