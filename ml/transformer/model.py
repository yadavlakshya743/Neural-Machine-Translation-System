"""Complete Sequence-to-Sequence Transformer Model for Neural Machine Translation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class TransformerSeq2Seq(nn.Module):
    """End-to-End Encoder-Decoder Transformer model for Neural Machine Translation."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 5000,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_encoder_layers,
            dropout=dropout,
            max_len=max_len,
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_decoder_layers,
            dropout=dropout,
            max_len=max_len,
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights with Xavier uniform distribution
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequence.
        Shape: (batch_size, 1, 1, src_len)
        """
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create combined causal (subsequent) and padding mask for target sequence.
        Shape: (batch_size, 1, tgt_len, tgt_len)
        """
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        # Triangular lower causal mask
        causal_mask = (
            torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(1)
        )
        return tgt_pad_mask & causal_mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        return self.decoder(
            tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for teacher-forcing training.

        Args:
            src: Source token IDs (batch_size, src_len)
            tgt: Target token IDs (batch_size, tgt_len)
            src_mask: Optional custom mask
            tgt_mask: Optional custom mask

        Returns:
            Logits of shape (batch_size, tgt_len, tgt_vocab_size)
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encoder(src, src_mask=src_mask)
        decoder_out = self.decoder(
            tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        return self.generator(decoder_out)
