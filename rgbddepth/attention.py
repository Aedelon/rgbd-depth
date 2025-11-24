#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Adaptive attention mechanisms with multiple backend support.
Supports xformers, torch, and manual implementations with automatic fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class AdaptiveCrossAttention(nn.Module):
    """Cross-attention module with pluggable backends.

    Supports three backends:
    - 'xformers': Memory-efficient attention (GPU only, requires xformers)
    - 'torch': Standard PyTorch MultiheadAttention (all devices)
    - 'manual': Custom memory-efficient implementation (good for MPS)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        backend: str = "torch",
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        """
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            backend: Backend to use ('xformers', 'torch', 'manual')
            dropout: Dropout probability
            batch_first: If True, expect (batch, seq, feature) format
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.backend = backend
        self.dropout = dropout
        self.batch_first = batch_first

        # Validate backend
        if backend == "xformers":
            try:
                import xformers.ops as xops
                self.xops = xops
            except ImportError:
                raise ImportError(
                    "xformers not available. Install with: pip install xformers"
                )

        # Initialize projection layers (shared across all backends)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # For torch backend, we'll use the projections manually
        # to maintain compatibility with xformers/manual backends
        if backend == "torch":
            # Create a dummy MultiheadAttention for reference
            # but we'll use our custom projections
            self._torch_mha = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=batch_first,
                bias=True
            )

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor (B, N_q, C) if batch_first else (N_q, B, C)
            key: Key tensor (B, N_k, C) if batch_first else (N_k, B, C)
            value: Value tensor (B, N_k, C) if batch_first else (N_k, B, C)
            attn_mask: Attention mask

        Returns:
            Tuple of (output, attention_weights)
            Note: attention_weights may be None for some backends
        """
        if self.backend == "xformers":
            return self._forward_xformers(query, key, value, attn_mask)
        elif self.backend == "torch":
            return self._forward_torch(query, key, value, attn_mask)
        elif self.backend == "manual":
            return self._forward_manual(query, key, value, attn_mask)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _forward_xformers(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass using xformers memory-efficient attention."""
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        if self.batch_first:
            B, N_q, C = q.shape
            _, N_k, _ = k.shape

            q = q.reshape(B, N_q, self.num_heads, self.head_dim)
            k = k.reshape(B, N_k, self.num_heads, self.head_dim)
            v = v.reshape(B, N_k, self.num_heads, self.head_dim)
        else:
            N_q, B, C = q.shape
            N_k, _, _ = k.shape

            q = q.reshape(N_q, B, self.num_heads, self.head_dim).transpose(0, 1)
            k = k.reshape(N_k, B, self.num_heads, self.head_dim).transpose(0, 1)
            v = v.reshape(N_k, B, self.num_heads, self.head_dim).transpose(0, 1)

        # xformers expects (B, N, H, D) format
        out = self.xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_mask,
            p=self.dropout if self.training else 0.0
        )

        # Reshape back
        if self.batch_first:
            out = out.reshape(B, N_q, C)
        else:
            out = out.transpose(0, 1).reshape(N_q, B, C)

        # Output projection
        out = self.out_proj(out)

        return out, None  # xformers doesn't return attention weights

    def _forward_torch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using PyTorch's MultiheadAttention."""
        # Use torch's MultiheadAttention with our projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Torch MHA expects specific format
        if not self.batch_first:
            q_mha, k_mha, v_mha = q, k, v
        else:
            # Convert to (N, B, C) for MHA
            q_mha = q.transpose(0, 1)
            k_mha = k.transpose(0, 1)
            v_mha = v.transpose(0, 1)

        # Compute attention using scaled dot-product
        out, attn_weights = self._scaled_dot_product_attention(
            q_mha, k_mha, v_mha, attn_mask
        )

        # Convert back if needed
        if self.batch_first:
            out = out.transpose(0, 1)

        # Output projection
        out = self.out_proj(out)

        return out, attn_weights

    def _forward_manual(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass using manual memory-efficient implementation.

        Optimized for MPS and cases where xformers is not available.
        """
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        if self.batch_first:
            B, N_q, C = q.shape
            _, N_k, _ = k.shape

            q = q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
            # Now: (B, H, N, D)
        else:
            N_q, B, C = q.shape
            N_k, _, _ = k.shape

            q = q.reshape(N_q, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k.reshape(N_k, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.reshape(N_k, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            # Now: (B, H, N, D)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        out = torch.matmul(attn, v)

        # Reshape back
        if self.batch_first:
            out = out.transpose(1, 2).reshape(B, N_q, C)
        else:
            out = out.permute(2, 0, 1, 3).reshape(N_q, B, C)

        # Output projection
        out = self.out_proj(out)

        return out, None  # Don't return attention weights to save memory

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        # q, k, v: (N, B, C)
        N_q, B, C = q.shape
        N_k = k.shape[0]

        # Reshape to (B, H, N, D)
        q = q.reshape(N_q, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(N_k, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(N_k, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape back to (N, B, C)
        out = out.permute(2, 0, 1, 3).reshape(N_q, B, C)

        return out, attn_weights


def create_cross_attention(
    embed_dim: int,
    num_heads: int,
    backend: str = "auto",
    device: str = "cuda",
) -> AdaptiveCrossAttention:
    """Factory function to create cross-attention with automatic backend selection.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        backend: Backend to use ('auto', 'xformers', 'torch', 'manual')
        device: Target device

    Returns:
        AdaptiveCrossAttention module with appropriate backend
    """
    device_type = device.split(":")[0]

    if backend == "auto":
        # Auto-select backend based on device
        if device_type == "cuda":
            # Try xformers, fall back to torch
            try:
                import xformers
                backend = "xformers"
                print(f"Using xformers backend for attention (GPU optimized)")
            except ImportError:
                backend = "torch"
                print(f"xformers not available, using torch backend")
        elif device_type == "mps":
            # MPS: use manual implementation (better than torch MHA on MPS)
            backend = "manual"
            print(f"Using manual attention backend (MPS optimized)")
        else:  # CPU
            backend = "torch"
            print(f"Using torch backend for attention")

    return AdaptiveCrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        backend=backend,
        batch_first=True
    )