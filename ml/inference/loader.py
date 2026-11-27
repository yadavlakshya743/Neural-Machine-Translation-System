"""Load a trained checkpoint together with the exact tokenizers it was trained with."""

import os
from typing import Any, Dict, Optional, Tuple

import torch

from ml.inference.translator import Translator
from ml.tokenization.tokenizer import SubwordTokenizer, file_sha256
from ml.transformer.model import TransformerSeq2Seq


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint and its tokenizers do not belong together (PRD TEST-006)."""


def _resolve(path: str, checkpoint_path: str) -> str:
    """Prefer the tokenizer copy saved next to the checkpoint (each run keeps its own, so a
    Kaggle-trained model works even if the local ml/tokenization files differ); fall back
    to the repo-relative path recorded at training time."""
    alt = os.path.join(os.path.dirname(checkpoint_path), os.path.basename(path))
    if path and os.path.exists(alt):
        return alt
    return path


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    src_tokenizer_path: Optional[str] = None,
    tgt_tokenizer_path: Optional[str] = None,
    verify_tokenizers: bool = True,
) -> Tuple[TransformerSeq2Seq, SubwordTokenizer, SubwordTokenizer, Dict[str, Any]]:
    """Returns (model, src_tokenizer, tgt_tokenizer, meta) for a checkpoint written by Trainer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        # mmap: weights are read from the file on demand instead of being held twice in RAM
        # while the state dict is copied into the model (matters on 512 MB hosts).
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False, mmap=True)
    except (RuntimeError, TypeError):  # older checkpoint formats / torch versions
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta") or {}
    model_cfg = meta.get("model_config")
    if not model_cfg:
        raise CheckpointMismatchError(
            f"{checkpoint_path} has no model_config metadata; it was written by an older "
            "version of the training script and cannot be loaded reliably. Retrain."
        )

    tok_meta = meta.get("tokenizers", {})
    src_path = src_tokenizer_path or _resolve(tok_meta.get("src", {}).get("path", ""), checkpoint_path)
    tgt_path = tgt_tokenizer_path or _resolve(tok_meta.get("tgt", {}).get("path", ""), checkpoint_path)
    for side, path in (("source", src_path), ("target", tgt_path)):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"{side} tokenizer not found: {path!r}")

    if verify_tokenizers:
        for side, path in (("src", src_path), ("tgt", tgt_path)):
            expected = tok_meta.get(side, {}).get("sha256")
            if expected and file_sha256(path) != expected:
                raise CheckpointMismatchError(
                    f"{side} tokenizer {path} differs from the one this checkpoint was trained "
                    "with (sha256 mismatch). Use the tokenizer files saved with the checkpoint."
                )

    src_tok = SubwordTokenizer(model_path=src_path)
    tgt_tok = SubwordTokenizer(model_path=tgt_path)
    if src_tok.vocab_size != model_cfg["src_vocab_size"] or tgt_tok.vocab_size != model_cfg["tgt_vocab_size"]:
        raise CheckpointMismatchError("Tokenizer vocabulary sizes do not match the checkpoint.")

    model = TransformerSeq2Seq.from_config(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, src_tok, tgt_tok, meta


def load_translator(checkpoint_path: str, device: str = "cpu", **translator_kwargs) -> Tuple[Translator, Dict[str, Any]]:
    model, src_tok, tgt_tok, meta = load_checkpoint(checkpoint_path, device=device)
    translator = Translator(
        model=model,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        device=device,
        src_lang=meta.get("src_lang"),
        max_src_len=meta.get("max_src_len", 128),
        **translator_kwargs,
    )
    return translator, meta
