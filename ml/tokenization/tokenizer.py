"""Subword Tokenizer Interface and Preprocessing for English and Hindi."""

from typing import List, Optional
import os


class SubwordTokenizer:
    """Wrapper for SentencePiece/BPE tokenizers with special token handling."""

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.sp = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str):
        """Loads a SentencePiece model file."""
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path

    @property
    def vocab_size(self) -> int:
        if self.sp is not None:
            return len(self.sp)
        return 16000

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text to a sequence of token IDs."""
        if self.sp is not None:
            ids = self.sp.encode_as_ids(text)
        else:
            # Fallback simple ASCII/character-based mock for testing when model file is not yet trained
            ids = [ord(c) % 15000 + 4 for c in text.strip()][:120]

        if add_special_tokens:
            return [self.SOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back to a text string."""
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.PAD_ID, self.SOS_ID, self.EOS_ID)]

        if self.sp is not None:
            return self.sp.decode_ids(ids)

        # Fallback decode
        try:
            return "".join([chr(i - 4) for i in ids if i >= 4])
        except Exception:
            return " ".join(map(str, ids))

    @staticmethod
    def train_spm(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 16000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ):
        """Trains a SentencePiece model on a text corpus file."""
        import sentencepiece as spm

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=SubwordTokenizer.PAD_ID,
            bos_id=SubwordTokenizer.SOS_ID,
            eos_id=SubwordTokenizer.EOS_ID,
            unk_id=SubwordTokenizer.UNK_ID,
        )
