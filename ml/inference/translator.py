"""High-level translation inference engine."""

import time
from typing import Dict, Any, Optional
import torch
from ml.transformer.model import TransformerSeq2Seq
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.decoding.strategy import GreedyDecoder, BeamSearchDecoder


class Translator:
    """High-level Translation interface managing model, tokenizers, and decoding strategies."""

    def __init__(
        self,
        model: TransformerSeq2Seq,
        src_tokenizer: SubwordTokenizer,
        tgt_tokenizer: SubwordTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_strategy: str = "beam_search",
        beam_size: int = 4,
        alpha: float = 0.6,
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.greedy_decoder = GreedyDecoder(
            model=self.model,
            sos_idx=SubwordTokenizer.SOS_ID,
            eos_idx=SubwordTokenizer.EOS_ID,
            pad_idx=SubwordTokenizer.PAD_ID,
        )

        self.beam_decoder = BeamSearchDecoder(
            model=self.model,
            sos_idx=SubwordTokenizer.SOS_ID,
            eos_idx=SubwordTokenizer.EOS_ID,
            pad_idx=SubwordTokenizer.PAD_ID,
            beam_size=beam_size,
            alpha=alpha,
        )

        self.default_strategy = default_strategy

    def translate(
        self,
        text: str,
        strategy: Optional[str] = None,
        beam_size: int = 4,
        alpha: float = 0.6,
        max_len: int = 128,
    ) -> Dict[str, Any]:
        """Translates a single source text string into the target language.

        Returns:
            Dictionary with translated_text, latency_ms, tokens_generated, and strategy_used.
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        # 1. Tokenize source
        src_ids = self.src_tokenizer.encode(text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # 2. Decode
        if strategy == "greedy":
            out_tensor = self.greedy_decoder.decode(src_tensor, max_len=max_len)
        else:
            out_tensor = self.beam_decoder.decode(
                src_tensor, max_len=max_len, beam_size=beam_size, alpha=alpha
            )

        # 3. Detokenize target
        tgt_ids = out_tensor.squeeze(0).tolist()
        translated_text = self.tgt_tokenizer.decode(tgt_ids, skip_special_tokens=True)

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "source_text": text,
            "translated_text": translated_text,
            "tokens_generated": len(tgt_ids),
            "latency_ms": round(latency_ms, 2),
            "strategy": strategy,
        }
