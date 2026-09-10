"""Scaled Dot-Product Attention and Multi-Head Attention mechanisms."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            q: Queries of shape (batch_size, n_heads, seq_len_q, d_k)
            k: Keys of shape (batch_size, n_heads, seq_len_k, d_k)
            v: Values of shape (batch_size, n_heads, seq_len_v, d_v)
            mask: Optional mask broadcastable to (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            context: (batch_size, n_heads, seq_len_q, d_v)
            attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask out positions where mask is 0 (or False) by filling with -1e9
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)

        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer allowing the model to jointly attend to information

    from different representation subspaces at different positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # 1. Project inputs linearly and split into n_heads: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = (
            self.w_q(q)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.w_k(k)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.w_v(v)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # 2. Apply attention
        context, attn_weights = self.attention(q, k, v, mask=mask)

        # 3. Concatenate heads and apply final linear projection
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        output = self.w_o(context)

        return self.dropout(output), attn_weights
