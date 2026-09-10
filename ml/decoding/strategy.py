"""Decoding Strategies for Transformer NMT (Greedy and Beam Search)."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import torch.nn.functional as F
from ml.transformer.model import TransformerSeq2Seq


class DecodingStrategy(ABC):
    """Abstract base class for sequence decoding strategies."""

    def __init__(self, model: TransformerSeq2Seq, sos_idx: int, eos_idx: int, pad_idx: int):
        self.model = model
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    @abstractmethod
    def decode(
        self,
        src: torch.Tensor,
        max_len: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        """Decodes source tokens into target sequence.

        Args:
            src: (1, src_len) source token IDs
            max_len: Maximum target generation length

        Returns:
            Tensor of generated token IDs (1, seq_len)
        """
        pass


class GreedyDecoder(DecodingStrategy):
    """Deterministic greedy search: at each timestep chooses the token with highest probability."""

    def decode(self, src: torch.Tensor, max_len: int = 128, **kwargs) -> torch.Tensor:
        self.model.eval()
        device = src.device

        with torch.no_grad():
            src_mask = self.model.make_src_mask(src)
            memory = self.model.encode(src, src_mask)

            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_len):
                tgt_mask = self.model.make_tgt_mask(ys)
                out = self.model.decoder(ys, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask)
                prob = self.model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()

                ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)

                if next_word == self.eos_idx:
                    break

            return ys


class BeamHypothesis:
    """Hypothesis tracking a sequence and its cumulative log probability score."""

    def __init__(self, tokens: List[int], log_prob: float):
        self.tokens = tokens
        self.log_prob = log_prob

    def length_penalty(self, alpha: float = 0.6) -> float:
        # Wu et al. (2016) length normalization: ((5 + len) / 6)^alpha
        length = len(self.tokens)
        penalty = ((5.0 + length) / 6.0) ** alpha
        return self.log_prob / penalty


class BeamSearchDecoder(DecodingStrategy):
    """Beam search decoding exploring top-k candidates with length normalization penalty."""

    def __init__(
        self,
        model: TransformerSeq2Seq,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int,
        beam_size: int = 4,
        alpha: float = 0.6,
    ):
        super().__init__(model, sos_idx, eos_idx, pad_idx)
        self.beam_size = beam_size
        self.alpha = alpha

    def decode(
        self,
        src: torch.Tensor,
        max_len: int = 128,
        beam_size: int = None,
        alpha: float = None,
        **kwargs,
    ) -> torch.Tensor:
        self.model.eval()
        device = src.device
        beam_size = beam_size or self.beam_size
        alpha = alpha if alpha is not None else self.alpha

        with torch.no_grad():
            src_mask = self.model.make_src_mask(src)
            memory = self.model.encode(src, src_mask)

            completed_hypotheses: List[BeamHypothesis] = []
            current_beams = [BeamHypothesis([self.sos_idx], 0.0)]

            for _ in range(max_len):
                candidates: List[BeamHypothesis] = []

                for beam in current_beams:
                    ys = torch.tensor([beam.tokens], dtype=torch.long, device=device)
                    tgt_mask = self.model.make_tgt_mask(ys)
                    out = self.model.decoder(
                        ys, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask
                    )
                    logits = self.model.generator(out[:, -1])
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                    topk_probs, topk_indices = torch.topk(log_probs, beam_size)

                    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                        cand = BeamHypothesis(beam.tokens + [idx], beam.log_prob + prob)
                        if idx == self.eos_idx:
                            completed_hypotheses.append(cand)
                        else:
                            candidates.append(cand)

                # Keep top beam_size non-terminated candidates
                candidates.sort(key=lambda h: h.length_penalty(alpha), reverse=True)
                current_beams = candidates[:beam_size]

                if not current_beams:
                    break

            # If no beam hit EOS, consider all remaining beams as finished
            all_hypotheses = completed_hypotheses + current_beams
            all_hypotheses.sort(key=lambda h: h.length_penalty(alpha), reverse=True)
            best_tokens = all_hypotheses[0].tokens

            return torch.tensor([best_tokens], dtype=torch.long, device=device)
