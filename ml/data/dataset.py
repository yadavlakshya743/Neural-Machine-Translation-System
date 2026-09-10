"""PyTorch Dataset and dynamic collate function for parallel translation data."""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ml.tokenization.tokenizer import SubwordTokenizer


class TranslationDataset(Dataset):
    """Dataset for parallel bilingual sentences."""

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
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = SubwordTokenizer.PAD_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates and dynamically pads source and target sequences to max length in the batch.

    Args:
        batch: List of (src_tensor, tgt_tensor)
        pad_idx: ID used for padding

    Returns:
        src_padded: (batch_size, max_src_len)
        tgt_padded: (batch_size, max_tgt_len)
    """
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
