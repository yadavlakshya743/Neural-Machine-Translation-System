"""PyTorch datasets, token-bucket batching and padding collation for parallel translation data."""

import os
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from ml.tokenization.tokenizer import SubwordTokenizer


class TranslationDataset(Dataset):
    """Dataset for parallel bilingual sentences, tokenized on the fly (fine for small sets)."""

    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_tokenizer: SubwordTokenizer,
        tgt_tokenizer: SubwordTokenizer,
    ):
        assert len(src_sentences) == len(tgt_sentences), "Source and target sizes must match."
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = self.src_tokenizer.encode(self.src_sentences[idx], add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_sentences[idx], add_special_tokens=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class EncodedCorpus:
    """Pre-tokenized parallel corpus stored as flat int32 arrays plus offsets.

    Holding 1.6M pairs as Python lists of ints costs several GB; flat numpy arrays cost
    roughly 0.5 GB and can be cached to disk so later runs skip tokenization entirely.
    Stored ids exclude <sos>/<eos>; those are added in `EncodedPairDataset`.
    """

    def __init__(self, src_ids, src_offsets, tgt_ids, tgt_offsets):
        self.src_ids = src_ids
        self.src_offsets = src_offsets
        self.tgt_ids = tgt_ids
        self.tgt_offsets = tgt_offsets

    def __len__(self) -> int:
        return len(self.src_offsets) - 1

    @property
    def src_lengths(self) -> np.ndarray:
        """Lengths including the two special tokens."""
        return np.diff(self.src_offsets) + 2

    @property
    def tgt_lengths(self) -> np.ndarray:
        return np.diff(self.tgt_offsets) + 2

    def pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s = self.src_ids[self.src_offsets[idx]: self.src_offsets[idx + 1]]
        t = self.tgt_ids[self.tgt_offsets[idx]: self.tgt_offsets[idx + 1]]
        return s, t

    @classmethod
    def from_texts(
        cls,
        src_lines: Sequence[str],
        tgt_lines: Sequence[str],
        src_tok: SubwordTokenizer,
        tgt_tok: SubwordTokenizer,
        max_len: int = 128,
        chunk_size: int = 20000,
    ) -> Tuple["EncodedCorpus", int]:
        """Tokenizes a corpus, dropping pairs whose length (with specials) exceeds max_len.

        Returns (corpus, number_of_dropped_pairs).
        """
        assert len(src_lines) == len(tgt_lines)
        src_chunks, tgt_chunks, src_lens, tgt_lens = [], [], [], []
        dropped = 0
        for start in range(0, len(src_lines), chunk_size):
            s_enc = src_tok.encode_batch(list(src_lines[start: start + chunk_size]))
            t_enc = tgt_tok.encode_batch(list(tgt_lines[start: start + chunk_size]))
            for s, t in zip(s_enc, t_enc):
                if not s or not t or len(s) + 2 > max_len or len(t) + 2 > max_len:
                    dropped += 1
                    continue
                src_chunks.append(np.asarray(s, dtype=np.int32))
                tgt_chunks.append(np.asarray(t, dtype=np.int32))
                src_lens.append(len(s))
                tgt_lens.append(len(t))

        def _flatten(chunks, lens):
            ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
            offsets = np.zeros(len(lens) + 1, dtype=np.int64)
            np.cumsum(lens, out=offsets[1:])
            return ids, offsets

        s_ids, s_off = _flatten(src_chunks, src_lens)
        t_ids, t_off = _flatten(tgt_chunks, tgt_lens)
        return cls(s_ids, s_off, t_ids, t_off), dropped

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, src_ids=self.src_ids, src_offsets=self.src_offsets,
                 tgt_ids=self.tgt_ids, tgt_offsets=self.tgt_offsets)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "EncodedCorpus":
        d = np.load(path)
        return cls(d["src_ids"], d["src_offsets"], d["tgt_ids"], d["tgt_offsets"])


class EncodedPairDataset(Dataset):
    """Serves (src, tgt) tensors from an EncodedCorpus, adding <sos>/<eos>."""

    def __init__(self, corpus: EncodedCorpus):
        self.corpus = corpus
        self._sos = np.array([SubwordTokenizer.SOS_ID], dtype=np.int64)
        self._eos = np.array([SubwordTokenizer.EOS_ID], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self.corpus.pair(idx)
        s = np.concatenate([self._sos, s, self._eos])
        t = np.concatenate([self._sos, t, self._eos])
        return torch.from_numpy(s), torch.from_numpy(t)


class TokenBucketBatchSampler(Sampler[List[int]]):
    """Groups sentences of similar length into batches capped by padded token count.

    Fixed-size batches waste most of their compute on padding. Here each batch holds as
    many sentences as fit in `max_tokens` (counting padding: n_sentences * longest
    sentence). Order is a deterministic function of (seed, epoch), which lets a resumed
    run skip exactly the batches it already trained on via `start_batch`.
    """

    def __init__(
        self,
        src_lengths: Sequence[int],
        tgt_lengths: Sequence[int],
        max_tokens: int,
        shuffle: bool = True,
        seed: int = 0,
        pool_factor: int = 100,
    ):
        self.lengths = np.maximum(np.asarray(src_lengths), np.asarray(tgt_lengths))
        if len(self.lengths) and self.lengths.max() > max_tokens:
            raise ValueError(
                f"max_tokens={max_tokens} is smaller than the longest sentence "
                f"({int(self.lengths.max())} tokens)."
            )
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.pool_factor = pool_factor
        self.epoch = 0
        self.start_batch = 0
        self._cache_epoch: Optional[int] = None
        self._cache: List[List[int]] = []

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = epoch
        self.start_batch = start_batch

    def _build(self) -> List[List[int]]:
        if self._cache_epoch == self.epoch:
            return self._cache
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n) if self.shuffle else np.arange(n)

        # Sort within large pools: near-uniform lengths per batch while keeping randomness.
        approx_batch = max(1, self.max_tokens // max(1, int(np.median(self.lengths)) if n else 1))
        pool = max(1, approx_batch * self.pool_factor)
        batches: List[List[int]] = []
        for p in range(0, n, pool):
            idx = order[p: p + pool]
            idx = idx[np.argsort(self.lengths[idx], kind="stable")]
            batch: List[int] = []
            longest = 0
            for i in idx:
                l = int(self.lengths[i])
                if batch and max(longest, l) * (len(batch) + 1) > self.max_tokens:
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(int(i))
                longest = max(longest, l)
            if batch:
                batches.append(batch)

        if self.shuffle:
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]
        self._cache_epoch, self._cache = self.epoch, batches
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._build()
        for b in batches[self.start_batch:]:
            yield b

    def __len__(self) -> int:
        return max(0, len(self._build()) - self.start_batch)


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = SubwordTokenizer.PAD_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates and dynamically pads source and target sequences to max length in the batch.

    Returns:
        src_padded: (batch_size, max_src_len)
        tgt_padded: (batch_size, max_tgt_len)
    """
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
