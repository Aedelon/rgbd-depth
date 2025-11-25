#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Flexible cross-attention module with xFormers support and automatic fallback."""

import torch
import torch.nn as nn


class FlexibleCrossAttention(nn.MultiheadAttention):
    """Cross-attention with optional xFormers support and automatic fallback to SDPA.

    This module inherits from nn.MultiheadAttention to ensure weight compatibility.
    It overrides forward() to use xFormers when available and requested.

    Uses:
    1. xFormers memory-efficient attention (CUDA only, if installed and use_xformers=True)
    2. PyTorch native SDPA (Scaled Dot Product Attention, PyTorch 2.0+, default)
    3. Standard MultiheadAttention (fallback for older PyTorch versions)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        use_xformers: Whether to attempt using xFormers (only works on CUDA)
    """

    def __init__(self, embed_dim, num_heads, use_xformers=False, **kwargs):
        # Initialize parent with batch_first=True to match original usage
        super().__init__(embed_dim, num_heads, batch_first=True, **kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Check if xFormers is available and requested
        self.use_xformers = use_xformers and self._check_xformers()

    def _check_xformers(self):
        """Check if xFormers is available for import.

        Returns:
            bool: True if xFormers can be imported, False otherwise
        """
        try:
            import importlib.util

            return importlib.util.find_spec("xformers.ops") is not None
        except (ImportError, ValueError):
            return False

    def forward(self, query, key, value, **kwargs):
        """Forward pass with automatic backend selection.

        Args:
            query: Query tensor of shape [B, N, C]
            key: Key tensor of shape [B, N, C]
            value: Value tensor of shape [B, N, C]

        Returns:
            tuple: (output, attention_weights)
                - output: Attention output of shape [B, N, C]
                - attention_weights: None (not computed for efficiency)
        """
        if not self.use_xformers:
            # Standard path using parent nn.MultiheadAttention (with SDPA in PyTorch 2.0+)
            # This uses the original weights (in_proj_weight, out_proj) from checkpoint
            return super().forward(query, key, value, need_weights=False, **kwargs)
        else:
            # xFormers memory-efficient attention path
            import xformers.ops as xops

            # Use parent's projection weights for Q, K, V
            # in_proj_weight contains concatenated [W_q; W_k; W_v]
            # This ensures we use the exact same weights as standard MultiheadAttention
            if self.in_proj_weight is not None:
                # Split the combined in_proj_weight into Q, K, V weights
                w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
                b_q, b_k, b_v = None, None, None
                if self.in_proj_bias is not None:
                    b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)

                # Apply projections using the same weights as standard attention
                q = torch.nn.functional.linear(query, w_q, b_q)
                k = torch.nn.functional.linear(key, w_k, b_k)
                v = torch.nn.functional.linear(value, w_v, b_v)
            else:
                # Separate projection weights (shouldn't happen with default config)
                q = torch.nn.functional.linear(query, self.q_proj_weight, self.in_proj_bias)
                k = torch.nn.functional.linear(key, self.k_proj_weight)
                v = torch.nn.functional.linear(value, self.v_proj_weight)

            # Reshape for multi-head attention: [B, N, C] -> [B, N, H, C//H]
            B, N, C = q.shape
            q = q.reshape(B, N, self.num_heads, self.head_dim)
            k = k.reshape(B, N, self.num_heads, self.head_dim)
            v = v.reshape(B, N, self.num_heads, self.head_dim)

            # Apply xFormers memory-efficient attention
            # This is significantly faster and uses less memory than standard attention
            out = xops.memory_efficient_attention(q, k, v)

            # Reshape back: [B, N, H, C//H] -> [B, N, C]
            out = out.reshape(B, N, C)

            # Use parent's output projection (same weights as standard attention)
            out = torch.nn.functional.linear(out, self.out_proj.weight, self.out_proj.bias)

            return out, None
