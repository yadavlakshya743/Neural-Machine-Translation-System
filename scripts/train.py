#!/usr/bin/env python3
"""Train the custom Transformer NMT model for one translation direction.

Grounding: Attention Is All You Need (Vaswani et al., NeurIPS 2017)
Data:      IIT Bombay English-Hindi Parallel Corpus (run scripts/prepare_data.py first)

Examples:
  python scripts/train.py --config ml/configs/baseline_config.yaml --direction en-hi
  python scripts/train.py --config ml/configs/transformer_base.yaml --direction hi-en --max_hours 11 --resume auto
"""

import argparse
import os
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data.dataset import (EncodedCorpus, EncodedPairDataset,  # noqa: E402
                             TokenBucketBatchSampler, pad_collate_fn)
from ml.evaluation.evaluator import TranslationEvaluator  # noqa: E402
from ml.inference.translator import Translator  # noqa: E402
from ml.tokenization.tokenizer import SubwordTokenizer  # noqa: E402
from ml.training.trainer import TrainConfig, Trainer  # noqa: E402
from ml.transformer.model import TransformerSeq2Seq  # noqa: E402


def read_lines(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run scripts/prepare_data.py first.")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def load_corpus(split, src, tgt, src_tok, tgt_tok, processed_dir, max_len, max_samples=0):
    """Tokenizes a split once and caches it, keyed by tokenizer hashes and filters."""
    key = f"{split}_{src}-{tgt}_{src_tok.sha256[:8]}{tgt_tok.sha256[:8]}_L{max_len}_N{max_samples or 'all'}"
    cache = os.path.join(processed_dir, "cache", key + ".npz")
    if os.path.exists(cache):
        corpus = EncodedCorpus.load(cache)
        print(f"[*] Loaded cached {split} corpus: {len(corpus):,} pairs ({cache})")
        return corpus
    src_lines = read_lines(os.path.join(processed_dir, f"{split}_{src}.txt"))
    tgt_lines = read_lines(os.path.join(processed_dir, f"{split}_{tgt}.txt"))
    assert len(src_lines) == len(tgt_lines), f"{split}: source/target line counts differ"
    if max_samples and len(src_lines) > max_samples:
        src_lines, tgt_lines = src_lines[:max_samples], tgt_lines[:max_samples]
    t0 = time.time()
    corpus, dropped = EncodedCorpus.from_texts(src_lines, tgt_lines, src_tok, tgt_tok, max_len=max_len)
    corpus.save(cache)
    print(f"[+] Tokenized {split}: kept {len(corpus):,}, dropped {dropped:,} over {max_len} tokens "
          f"({time.time() - t0:.0f}s)")
    return corpus


def main():
    p = argparse.ArgumentParser(description="Train the custom Transformer for English <-> Hindi NMT.")
    p.add_argument("--config", default="ml/configs/baseline_config.yaml")
    p.add_argument("--direction", choices=["en-hi", "hi-en"], default=None,
                   help="Overrides data.direction in the config.")
    p.add_argument("--run_name", default=None, help="Default: <model.name>_<direction>")
    p.add_argument("--resume", default=None,
                   help="'auto' resumes <run_dir>/last.pt if it exists, or give a checkpoint path.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_tokens", type=int, default=None, help="Tokens per micro-batch (incl. padding).")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None, help="0 = all training pairs.")
    p.add_argument("--lr", type=float, default=None, help="Peak learning rate.")
    p.add_argument("--max_hours", type=float, default=None, help="Wall-clock budget for this session.")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m, d, t = cfg["model"], cfg["data"], cfg["training"]

    direction = args.direction or d.get("direction", "en-hi")
    src_lang, tgt_lang = direction.split("-")
    run_name = args.run_name or f"{m['name']}_{direction}"
    run_dir = os.path.join(t.get("checkpoint_dir", "models"), run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Apply CLI overrides back into cfg so the saved config is exactly what ran.
    if args.epochs is not None:
        t["epochs"] = args.epochs
    if args.max_tokens is not None:
        d["max_tokens"] = args.max_tokens
    if args.accum_steps is not None:
        t["accum_steps"] = args.accum_steps
    if args.max_train_samples is not None:
        d["max_train_samples"] = args.max_train_samples
    if args.lr is not None:
        t["learning_rate"] = args.lr
    if args.no_amp:
        t["use_amp"] = False
    d["direction"] = direction

    seed = t.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f" Training {run_name} | direction {src_lang}->{tgt_lang} | device {device}")
    if device.startswith("cuda"):
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    # Tokenizers: one per language, shared by both directions.
    tok_dir = d.get("tokenizer_dir", "ml/tokenization")
    # Forward slashes so checkpoints trained on Windows load on Linux servers too.
    tok_path = {lang: f"{tok_dir}/spm_{lang}.model".replace("\\", "/") for lang in ("en", "hi")}
    for path in tok_path.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")
    src_tok = SubwordTokenizer(tok_path[src_lang])
    tgt_tok = SubwordTokenizer(tok_path[tgt_lang])
    for path in tok_path.values():  # keep tokenizers next to the checkpoint
        shutil.copy2(path, os.path.join(run_dir, os.path.basename(path)))

    max_len = d.get("max_len", 128)
    processed = d.get("processed_dir", "data/processed")
    train_corpus = load_corpus("train", src_lang, tgt_lang, src_tok, tgt_tok, processed, max_len,
                               d.get("max_train_samples") or 0)
    dev_corpus = load_corpus("dev", src_lang, tgt_lang, src_tok, tgt_tok, processed, max_len)

    max_tokens = d.get("max_tokens", 4096)
    train_sampler = TokenBucketBatchSampler(train_corpus.src_lengths, train_corpus.tgt_lengths,
                                            max_tokens=max_tokens, shuffle=True, seed=seed)
    dev_sampler = TokenBucketBatchSampler(dev_corpus.src_lengths, dev_corpus.tgt_lengths,
                                          max_tokens=max_tokens, shuffle=False)
    pin = device.startswith("cuda")
    train_loader = DataLoader(EncodedPairDataset(train_corpus), batch_sampler=train_sampler,
                              collate_fn=pad_collate_fn, num_workers=d.get("num_workers", 0), pin_memory=pin)
    dev_loader = DataLoader(EncodedPairDataset(dev_corpus), batch_sampler=dev_sampler,
                            collate_fn=pad_collate_fn, num_workers=0, pin_memory=pin)

    model = TransformerSeq2Seq(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_encoder_layers=m["n_encoder_layers"],
        n_decoder_layers=m["n_decoder_layers"],
        d_ff=m["d_ff"],
        dropout=m["dropout"],
        pad_idx=SubwordTokenizer.PAD_ID,
        max_len=max(512, max_len),
        tie_embeddings=m.get("tie_embeddings", False),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model: {n_params:,} trainable parameters | {len(train_sampler):,} batches/epoch "
          f"of <= {max_tokens} tokens")

    # Dev-set BLEU/chrF++ after each epoch, used for checkpoint selection.
    dev_src = read_lines(os.path.join(processed, f"dev_{src_lang}.txt"))
    dev_ref = read_lines(os.path.join(processed, f"dev_{tgt_lang}.txt"))

    def eval_fn(current_model):
        tr = Translator(current_model, src_tok, tgt_tok, device=device, src_lang=src_lang, max_src_len=max_len)
        hyps = [tr.detokenize(ids) for ids in tr.translate_batch(dev_src, strategy="greedy", max_len=max_len)]
        scores = TranslationEvaluator.compute_metrics(hyps, [[r] for r in dev_ref])
        print(f"    dev sample: {dev_src[0]!r}\n             -> {hyps[0]!r}")
        return {"dev_bleu": scores["bleu"], "dev_chrf2": scores["chrf2"]}

    meta = {
        "run_name": run_name,
        "direction": direction,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "max_src_len": max_len,
        "dataset": "IIT Bombay English-Hindi Parallel Corpus (official split)",
        "train_pairs": len(train_corpus),
        "dev_pairs": len(dev_corpus),
        "total_parameters": n_params,
        "tokenizers": {
            "src": {"path": tok_path[src_lang], "sha256": src_tok.sha256, "vocab_size": src_tok.vocab_size},
            "tgt": {"path": tok_path[tgt_lang], "sha256": tgt_tok.sha256, "vocab_size": tgt_tok.vocab_size},
        },
        "config": cfg,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
    }
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    train_cfg = TrainConfig(
        learning_rate=t.get("learning_rate", 7e-4),
        warmup_steps=t.get("warmup_steps", 4000),
        label_smoothing=t.get("label_smoothing", 0.1),
        grad_clip=t.get("grad_clip", 1.0),
        weight_decay=t.get("weight_decay", 0.0001),
        accum_steps=t.get("accum_steps", 1),
        use_amp=t.get("use_amp", True),
        epochs=t.get("epochs", 10),
        patience=t.get("patience", 3),
        select_metric=t.get("select_metric", "bleu"),
        save_every_steps=t.get("save_every_steps", 1000),
        log_interval=t.get("log_interval", 100),
        max_hours=args.max_hours,
    )
    trainer = Trainer(model, train_loader, dev_loader, config=train_cfg, device=device,
                      run_dir=run_dir, meta=meta, eval_fn=eval_fn)

    resume_path = None
    if args.resume == "auto":
        resume_path = trainer.last_path if os.path.exists(trainer.last_path) else None
    elif args.resume:
        resume_path = args.resume
    if resume_path:
        trainer.resume(resume_path)
        trainer.cfg.max_hours = args.max_hours  # budget is per session, not inherited

    status = trainer.fit()
    print("=" * 72)
    print(f"[+] Training status: {status}")
    print(f"[+] Best {train_cfg.select_metric}: {trainer.state.best_score} (epoch {trainer.state.best_epoch})")
    print(f"[+] Artifacts in {run_dir}: best.pt (inference), last.pt (resume), metrics.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
