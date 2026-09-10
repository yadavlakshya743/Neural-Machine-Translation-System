"""Unit tests for custom PyTorch Transformer Seq2Seq components."""

import unittest

try:
    import torch
    from ml.transformer.embedding import PositionalEncoding, TransformerEmbedding
    from ml.transformer.attention import MultiHeadAttention, ScaledDotProductAttention
    from ml.transformer.encoder import TransformerEncoder
    from ml.transformer.decoder import TransformerDecoder
    from ml.transformer.model import TransformerSeq2Seq
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for Transformer tests")
class TestTransformerArchitecture(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len_src = 10
        self.seq_len_tgt = 8
        self.d_model = 64
        self.n_heads = 4
        self.src_vocab = 100
        self.tgt_vocab = 120

    def test_positional_encoding_shape(self):
        pe = PositionalEncoding(d_model=self.d_model, max_len=100)
        x = torch.zeros(self.batch_size, self.seq_len_src, self.d_model)
        out = pe(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len_src, self.d_model))

    def test_multi_head_attention_shape(self):
        mha = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads)
        q = torch.randn(self.batch_size, self.seq_len_tgt, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len_src, self.d_model)
        v = torch.randn(self.batch_size, self.seq_len_src, self.d_model)

        out, weights = mha(q, k, v)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len_tgt, self.d_model))
        self.assertEqual(
            weights.shape, (self.batch_size, self.n_heads, self.seq_len_tgt, self.seq_len_src)
        )

    def test_transformer_seq2seq_forward(self):
        model = TransformerSeq2Seq(
            src_vocab_size=self.src_vocab,
            tgt_vocab_size=self.tgt_vocab,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=128,
            dropout=0.0,
        )

        src = torch.randint(1, self.src_vocab, (self.batch_size, self.seq_len_src))
        tgt = torch.randint(1, self.tgt_vocab, (self.batch_size, self.seq_len_tgt))

        logits = model(src, tgt)
        self.assertEqual(
            logits.shape, (self.batch_size, self.seq_len_tgt, self.tgt_vocab)
        )

    def test_causal_mask_structure(self):
        model = TransformerSeq2Seq(
            src_vocab_size=self.src_vocab,
            tgt_vocab_size=self.tgt_vocab,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )
        tgt = torch.tensor([[1, 2, 3, 0]])
        mask = model.make_tgt_mask(tgt)
        self.assertEqual(mask.shape, (1, 1, 4, 4))
        self.assertFalse(mask[0, 0, 0, 1].item())
        self.assertTrue(mask[0, 0, 0, 0].item())


if __name__ == "__main__":
    unittest.main()
