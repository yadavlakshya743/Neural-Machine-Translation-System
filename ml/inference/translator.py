"""High-level translation inference engine."""

import time
from typing import Any, Dict, List, Optional

import torch

from ml.data.preprocess import TextPreprocessor
from ml.decoding.strategy import BeamSearchDecoder, GreedyDecoder
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.transformer.model import TransformerSeq2Seq


class Translator:
    """Translation interface managing model, tokenizers, and decoding strategies."""

    def __init__(
        self,
        model: TransformerSeq2Seq,
        src_tokenizer: SubwordTokenizer,
        tgt_tokenizer: SubwordTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_strategy: str = "beam_search",
        beam_size: int = 4,
        alpha: float = 0.6,
        src_lang: Optional[str] = None,
        max_src_len: int = 128,
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.max_src_len = max_src_len

        ids = dict(
            sos_idx=SubwordTokenizer.SOS_ID,
            eos_idx=SubwordTokenizer.EOS_ID,
            pad_idx=SubwordTokenizer.PAD_ID,
        )
        self.greedy_decoder = GreedyDecoder(model=self.model, **ids)
        self.beam_decoder = BeamSearchDecoder(model=self.model, beam_size=beam_size, alpha=alpha, **ids)
        self.default_strategy = default_strategy
        self.default_beam_size = beam_size
        self.default_alpha = alpha

    def _normalize(self, text: str) -> str:
        # Apply the same normalization used when the training corpus was prepared (FR-003).
        if self.src_lang == "hi":
            return TextPreprocessor.normalize_hindi(text)
        if self.src_lang == "en":
            return TextPreprocessor.normalize_english(text)
        return text.strip()

    def _encode_src(self, texts: List[str]) -> List[List[int]]:
        out = []
        for ids in self.src_tokenizer.encode_batch([self._normalize(t) for t in texts]):
            ids = ids[: self.max_src_len - 2]  # bound input length (PRD security/reliability)
            out.append([SubwordTokenizer.SOS_ID] + ids + [SubwordTokenizer.EOS_ID])
        return out

    def translate_batch(
        self,
        texts: List[str],
        strategy: Optional[str] = None,
        beam_size: Optional[int] = None,
        alpha: Optional[float] = None,
        max_len: int = 128,
        batch_size: int = 64,
    ) -> List[List[int]]:
        """Translates many texts; returns generated target token ids per input (input order)."""
        strategy = strategy or self.default_strategy
        beam_size = beam_size or self.default_beam_size
        alpha = self.default_alpha if alpha is None else alpha

        encoded = self._encode_src(texts)
        # Sort by length so each batch has little padding, then restore input order.
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        results: List[Optional[List[int]]] = [None] * len(encoded)
        for start in range(0, len(order), batch_size):
            idx = order[start: start + batch_size]
            longest = max(len(encoded[i]) for i in idx)
            src = torch.full((len(idx), longest), SubwordTokenizer.PAD_ID, dtype=torch.long)
            for row, i in enumerate(idx):
                src[row, : len(encoded[i])] = torch.tensor(encoded[i])
            src = src.to(self.device)
            if strategy == "greedy":
                outs = self.greedy_decoder.decode(src, max_len=max_len)
            else:
                outs = self.beam_decoder.decode(src, max_len=max_len, beam_size=beam_size, alpha=alpha)
            for i, ids in zip(idx, outs):
                results[i] = ids
        return results

    def detokenize(self, ids: List[int]) -> str:
        return self.tgt_tokenizer.decode(ids, skip_special_tokens=True)

    def translate(
        self,
        text: str,
        strategy: Optional[str] = None,
        beam_size: int = 4,
        alpha: float = 0.6,
        max_len: int = 128,
    ) -> Dict[str, Any]:
        """Translates a single source string.

        Returns:
            Dictionary with translated_text, latency_ms, tokens_generated, and strategy.
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        ids = self.translate_batch([text], strategy=strategy, beam_size=beam_size,
                                   alpha=alpha, max_len=max_len)[0]
        return {
            "source_text": text,
            "translated_text": self.detokenize(ids),
            "tokens_generated": len(ids),
            "latency_ms": round((time.time() - start_time) * 1000.0, 2),
            "strategy": strategy,
        }
