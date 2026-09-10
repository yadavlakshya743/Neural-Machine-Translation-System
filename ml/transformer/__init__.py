"""Custom Transformer Seq2Seq Architecture for Neural Machine Translation.
Grounded in Vaswani et al. (2017) 'Attention Is All You Need'.
"""

from .attention import MultiHeadAttention, ScaledDotProductAttention
from .embedding import PositionalEncoding, TransformerEmbedding
from .encoder import TransformerEncoder, TransformerEncoderLayer
from .decoder import TransformerDecoder, TransformerDecoderLayer
from .model import TransformerSeq2Seq

__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "TransformerEmbedding",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "TransformerSeq2Seq",
]
