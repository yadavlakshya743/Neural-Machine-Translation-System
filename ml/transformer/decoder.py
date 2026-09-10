"""Transformer Decoder layers and decoder stack."""

from typing import Optional
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding
from .encoder import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """Single layer of Transformer Decoder consisting of:

    1. Masked Self-Attention (causal look-ahead mask)
    2. Cross-Attention (queries from decoder, keys & values from encoder memory)
    3. Position-wise Feed-Forward Network
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked self-attention
        self_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_out))

        # 2. Cross-attention over encoder memory
        cross_out, _ = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_out))

        # 3. Positionwise Feed-Forward
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_out))

        return tgt


class TransformerDecoder(nn.Module):
    """Transformer Decoder composed of an embedding module and a stack of N decoder layers."""

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
                TransformerDecoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt)
        for layer in self.layers:
            x = layer(x, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x)
