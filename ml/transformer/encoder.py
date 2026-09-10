"""Transformer Encoder layers and encoder stack."""

from typing import Optional
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single layer of Transformer Encoder consisting of Self-Attention and FFN

    with LayerNorm and Residual Connections.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN or Post-LN: here we follow the standard Post-LN from Vaswani et al.
        attn_out, _ = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_out))

        ff_out = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_out))

        return src


class TransformerEncoder(nn.Module):
    """Transformer Encoder composed of an embedding module and a stack of N encoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)
