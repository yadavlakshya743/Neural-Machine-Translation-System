"""Corpus normalization, Devanagari cleaning and filtering utilities."""

import re
import unicodedata
from typing import Tuple, Optional


class TextPreprocessor:
    """Preprocesses English and Hindi sentences for Neural Machine Translation."""

    # Devanagari Unicode range: \u0900-\u097F
    DEV_RANGE = re.compile(r"[\u0900-\u097F]")
    MULTIPLE_SPACES = re.compile(r"\s+")
    HTML_TAGS = re.compile(r"<[^>]+>")

    @classmethod
    def normalize_english(cls, text: str) -> str:
        """Normalizes English text (NFKC, lowercase strip, whitespace reduction)."""
        text = unicodedata.normalize("NFKC", text)
        text = cls.HTML_TAGS.sub(" ", text)
        text = text.replace("\t", " ").replace("\n", " ")
        text = cls.MULTIPLE_SPACES.sub(" ", text).strip()
        return text

    @classmethod
    def normalize_hindi(cls, text: str) -> str:
        """Normalizes Hindi text (Devanagari NFKC, nukta consolidation, punctuation)."""
        text = unicodedata.normalize("NFKC", text)
        text = cls.HTML_TAGS.sub(" ", text)
        # Standardize Devanagari Danda (। -> |) or maintain standard purna viram
        text = text.replace("\t", " ").replace("\n", " ")
        text = cls.MULTIPLE_SPACES.sub(" ", text).strip()
        return text

    @classmethod
    def clean_pair(
        cls, src: str, tgt: str, max_len: int = 128, max_ratio: float = 2.5
    ) -> Optional[Tuple[str, str]]:
        """Cleans and validates an aligned sentence pair.
        Rejects pairs that are empty, too long, or have extreme length ratios.
        """
        src = cls.normalize_english(src)
        tgt = cls.normalize_hindi(tgt)

        if not src or not tgt:
            return None

        # Identical text check
        if src.lower() == tgt.lower():
            return None

        src_tokens = src.split()
        tgt_tokens = tgt.split()

        if len(src_tokens) == 0 or len(tgt_tokens) == 0:
            return None

        if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
            return None

        ratio = max(len(src_tokens), len(tgt_tokens)) / min(len(src_tokens), len(tgt_tokens))
        if ratio > max_ratio:
            return None

        return src, tgt
