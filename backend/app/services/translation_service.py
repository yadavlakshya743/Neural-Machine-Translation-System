"""Translation service: loads trained checkpoints once per process and runs inference.

There is deliberately no placeholder/fake output: if no trained model is available for a
direction, requests fail with MODEL_NOT_AVAILABLE (PRD: "model before wrapper").
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from backend.app.core.config import settings

logger = logging.getLogger("nmt.service")

DIRECTIONS = ("en-hi", "hi-en")


class ModelNotAvailableError(RuntimeError):
    """The requested translation direction has no trained model loaded."""


class TranslationService:
    def __init__(self):
        self.translators: Dict[str, Any] = {}
        self.metas: Dict[str, Dict[str, Any]] = {}
        self.paths: Dict[str, str] = {}
        self.device = "cpu"
        self.load_errors: Dict[str, str] = {}

    @property
    def is_ready(self) -> bool:
        return bool(self.translators)

    def checkpoint_paths(self) -> Dict[str, str]:
        """Checkpoint per direction. models/serving.json (written by scripts/publish_models.py)
        overrides the .env paths, so a newer model can be served without a config change."""
        paths = {"en-hi": settings.MODEL_CHECKPOINT_PATH, "hi-en": settings.MODEL_CHECKPOINT_PATH_HI_EN}
        manifest = os.path.join(settings.MODELS_DIR, "serving.json")
        if os.path.exists(manifest):
            try:
                with open(manifest, encoding="utf-8") as f:
                    for direction, rel in json.load(f).items():
                        if direction in paths:
                            paths[direction] = os.path.join(settings.MODELS_DIR, rel)
            except (OSError, ValueError) as e:
                logger.error("serving_manifest_invalid path=%s error=%s", manifest, e)
        return paths

    def initialize(self) -> None:
        """Loads every configured checkpoint (model + its own tokenizers)."""
        import torch
        from ml.inference.loader import load_translator

        self.translators, self.metas, self.paths, self.load_errors = {}, {}, {}, {}
        self.device = "cuda" if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        for direction, path in self.checkpoint_paths().items():
            if not path or not os.path.exists(path):
                self.load_errors[direction] = f"checkpoint not found: {path}"
                continue
            try:
                translator, meta = load_translator(
                    path, device=self.device,
                    beam_size=settings.DEFAULT_BEAM_SIZE, alpha=settings.DEFAULT_LENGTH_PENALTY,
                )
            except Exception as e:  # CHECKPOINT_LOAD_FAILED: log, keep serving other directions
                self.load_errors[direction] = f"{type(e).__name__}: {e}"
                logger.error("checkpoint_load_failed direction=%s path=%s error=%s", direction, path, e)
                continue
            if meta.get("direction") != direction:
                self.load_errors[direction] = f"{path} is a {meta.get('direction')} model"
                logger.error("checkpoint_direction_mismatch direction=%s path=%s", direction, path)
                continue
            self.translators[direction] = translator
            self.metas[direction] = meta
            self.paths[direction] = path
            logger.info("model_loaded direction=%s run=%s path=%s device=%s",
                        direction, meta.get("run_name"), path, self.device)

    def translate(self, text: str, source_lang: str, target_lang: str, strategy: str,
                  beam_size: int, alpha: float, max_len: int) -> Dict[str, Any]:
        direction = f"{source_lang}-{target_lang}"
        translator = self.translators.get(direction)
        if translator is None:
            raise ModelNotAvailableError(f"No trained {source_lang}->{target_lang} model is loaded yet.")
        result = translator.translate(text=text, strategy=strategy, beam_size=beam_size,
                                      alpha=alpha, max_len=min(max_len, settings.MAX_GEN_LENGTH))
        result["model_version"] = self.metas[direction].get("run_name", "unknown")
        return result

    def model_info(self, direction: str) -> Dict[str, Any]:
        meta = self.metas[direction]
        cfg = meta.get("model_config", {})
        info: Dict[str, Any] = {
            "direction": direction,
            "run_name": meta.get("run_name", "unknown"),
            "parameters": meta.get("total_parameters"),
            "train_pairs": meta.get("train_pairs"),
            "checkpoint": self.paths[direction],
            "device": self.device,
            "tokenizer_sha256": {side: meta.get("tokenizers", {}).get(side, {}).get("sha256")
                                 for side in ("src", "tgt")},
        }
        for key in ("d_model", "n_heads", "n_encoder_layers", "n_decoder_layers", "d_ff",
                    "src_vocab_size", "tgt_vocab_size"):
            info[key] = cfg.get(key)
        metrics_path = os.path.join(os.path.dirname(self.paths[direction]), "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                state = json.load(f).get("state", {})
            info["trained_epochs"] = len(state.get("history", {}).get("epochs", []))
            info["best_dev_bleu"] = state.get("best_score")
        return info


translation_service = TranslationService()
