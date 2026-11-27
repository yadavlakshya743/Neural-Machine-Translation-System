"""Decoding strategies for Transformer NMT (batched Greedy and Beam Search).

Both strategies decode a whole batch of source sentences at once and return, for each
sentence, the generated target token ids *without* <sos> and cut before <eos>.
"""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn.functional as F

from ml.transformer.model import TransformerSeq2Seq


def length_penalty(length: int, alpha: float) -> float:
    """Wu et al. (2016) length normalization: ((5 + |Y|) / 6) ** alpha."""
    return ((5.0 + length) / 6.0) ** alpha


class DecodingStrategy(ABC):
    """Abstract base class for sequence decoding strategies (Strategy pattern, ADR-004)."""

    def __init__(self, model: TransformerSeq2Seq, sos_idx: int, eos_idx: int, pad_idx: int):
        self.model = model
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    @abstractmethod
    def decode(self, src: torch.Tensor, max_len: int = 128, **kwargs) -> List[List[int]]:
        """Decodes a padded batch of source ids (batch, src_len) into target id lists."""

    def _strip(self, ids: List[int]) -> List[int]:
        out = []
        for i in ids:
            if i == self.eos_idx:
                break
            if i not in (self.sos_idx, self.pad_idx):
                out.append(i)
        return out


class GreedyDecoder(DecodingStrategy):
    """Deterministic greedy search: at each step choose the highest-probability token."""

    @torch.no_grad()
    def decode(self, src: torch.Tensor, max_len: int = 128, **kwargs) -> List[List[int]]:
        self.model.eval()
        device = src.device
        batch = src.size(0)

        src_mask = self.model.make_src_mask(src)
        memory = self.model.encode(src, src_mask)

        ys = torch.full((batch, 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            out = self.model.decode(ys, memory, memory_mask=src_mask)
            logits = self.model.generator(out[:, -1])
            next_tok = logits.argmax(dim=-1)
            # Sentences that already produced <eos> only receive padding.
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_idx), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok == self.eos_idx
            if finished.all():
                break

        return [self._strip(row) for row in ys.tolist()]


class BeamSearchDecoder(DecodingStrategy):
    """Batched beam search with Wu et al. (2016) length normalization.

    All (sentence, beam) pairs are decoded as one tensor batch of size batch * beam_size.
    A sentence stops expanding once it has `beam_size` finished hypotheses.
    """

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

    @torch.no_grad()
    def decode(
        self,
        src: torch.Tensor,
        max_len: int = 128,
        beam_size: int = None,
        alpha: float = None,
        **kwargs,
    ) -> List[List[int]]:
        self.model.eval()
        device = src.device
        k = beam_size or self.beam_size
        alpha = self.alpha if alpha is None else alpha
        batch = src.size(0)

        src_mask = self.model.make_src_mask(src)
        memory = self.model.encode(src, src_mask)
        # Repeat each sentence k times: rows [b*k, ..., b*k + k - 1] belong to sentence b.
        memory = memory.repeat_interleave(k, dim=0)
        src_mask = src_mask.repeat_interleave(k, dim=0)

        seqs = torch.full((batch * k, 1), self.sos_idx, dtype=torch.long, device=device)
        scores = torch.zeros(batch, k, device=device)
        scores[:, 1:] = float("-inf")  # only the first beam is live at step 0

        finished: List[List[tuple]] = [[] for _ in range(batch)]  # (normalized score, tokens)
        done = [False] * batch
        batch_arange = torch.arange(batch, device=device).unsqueeze(1)

        for step in range(max_len):
            out = self.model.decode(seqs, memory, memory_mask=src_mask)
            log_probs = F.log_softmax(self.model.generator(out[:, -1]).float(), dim=-1)
            vocab = log_probs.size(-1)

            cand = (scores.view(-1, 1) + log_probs).view(batch, k * vocab)
            # 2k candidates guarantee at least k that do not end in <eos>
            # (each live beam contributes at most one <eos> candidate).
            top_scores, top_idx = cand.topk(min(2 * k, cand.size(1)), dim=1)
            beam_idx = top_idx // vocab
            tok_idx = top_idx % vocab
            is_eos = tok_idx == self.eos_idx

            # Record finished hypotheses: only <eos> candidates ranked within the top k.
            eos_pos = (is_eos[:, :k]).nonzero(as_tuple=False).tolist()
            gen_len = step + 1
            for b, j in eos_pos:
                if done[b]:
                    continue
                raw = top_scores[b, j].item()
                if raw == float("-inf"):
                    continue
                src_row = b * k + beam_idx[b, j].item()
                tokens = seqs[src_row, 1:].tolist()
                finished[b].append((raw / length_penalty(gen_len, alpha), tokens))
                if len(finished[b]) >= k:
                    done[b] = True

            if all(done):
                break

            # Keep the best k non-<eos> candidates as the new live beams.
            masked = top_scores.masked_fill(is_eos, float("-inf"))
            new_scores, pos = masked.topk(k, dim=1)
            new_beam = beam_idx.gather(1, pos)
            new_tok = tok_idx.gather(1, pos)

            prev = seqs.view(batch, k, -1)[batch_arange, new_beam]  # (batch, k, t)
            seqs = torch.cat([prev, new_tok.unsqueeze(-1)], dim=-1).view(batch * k, -1)
            scores = new_scores
            # Finished sentences keep decoding (cheap) but are ignored from here on.

        results: List[List[int]] = []
        final_len = seqs.size(1) - 1
        for b in range(batch):
            hyps = list(finished[b])
            if not hyps:  # nothing reached <eos> within max_len: fall back to live beams
                for j in range(k):
                    raw = scores[b, j].item()
                    if raw != float("-inf"):
                        hyps.append((raw / length_penalty(final_len, alpha),
                                     seqs[b * k + j, 1:].tolist()))
            best = max(hyps, key=lambda h: h[0])[1] if hyps else []
            results.append(self._strip(best))
        return results
