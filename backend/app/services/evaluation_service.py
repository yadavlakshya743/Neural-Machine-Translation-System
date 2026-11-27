"""Reads experiment results written by scripts/train.py and scripts/evaluate.py.

Each run folder under MODELS_DIR contains metrics.json (training/validation history) and
optionally eval_<split>_<strategy>.json reports. This powers the evaluation dashboard.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings

MAX_CURVE_POINTS = 200


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _downsample(points: List[Dict[str, Any]], n: int = MAX_CURVE_POINTS) -> List[Dict[str, Any]]:
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


def list_experiments() -> List[Dict[str, Any]]:
    runs = []
    for metrics_path in sorted(glob.glob(os.path.join(settings.MODELS_DIR, "*", "metrics.json"))):
        run_dir = os.path.dirname(metrics_path)
        data = _read_json(metrics_path)
        if not data:
            continue
        meta, state, cfg = data.get("meta", {}), data.get("state", {}), data.get("train_config", {})
        model_cfg = meta.get("config", {}).get("model", {})
        evaluations = []
        for path in sorted(glob.glob(os.path.join(run_dir, "eval_*.json"))):
            rep = _read_json(path)
            if not rep:
                continue
            evaluations.append({
                "split": rep.get("split"),
                "decoding": rep.get("decoding"),
                "bleu": rep.get("metrics", {}).get("bleu"),
                "chrf2": rep.get("metrics", {}).get("chrf2"),
                "bleu_signature": rep.get("metrics", {}).get("bleu_signature"),
                "samples_evaluated": rep.get("samples_evaluated"),
                "ms_per_sentence": rep.get("performance", {}).get("ms_per_sentence"),
                "examples": rep.get("samples", [])[:8],
                "timestamp": rep.get("timestamp"),
            })
        runs.append({
            "run_name": meta.get("run_name", os.path.basename(run_dir)),
            "direction": meta.get("direction"),
            "status": state.get("status"),
            "global_step": state.get("global_step"),
            "best_dev_bleu": state.get("best_score"),
            "best_epoch": state.get("best_epoch"),
            "hours_trained": round((state.get("elapsed_sec") or 0) / 3600, 2),
            "parameters": meta.get("total_parameters"),
            "train_pairs": meta.get("train_pairs"),
            "model": {k: model_cfg.get(k) for k in ("d_model", "n_heads", "n_encoder_layers",
                                                     "n_decoder_layers", "d_ff", "dropout")},
            "training": {k: cfg.get(k) for k in ("learning_rate", "warmup_steps", "label_smoothing",
                                                  "accum_steps", "epochs")},
            "epochs": state.get("history", {}).get("epochs", []),
            "train_curve": _downsample(state.get("history", {}).get("train", [])),
            "evaluations": evaluations,
        })
    return runs


def data_report() -> Optional[Dict[str, Any]]:
    return _read_json(settings.DATA_REPORT_PATH)
