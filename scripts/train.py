#!/usr/bin/env python3
"""CLI training script for Transformer English <-> Hindi NMT.

Grounding: Attention Is All You Need (Vaswani et al., NeurIPS 2017)
Trained on: IIT Bombay English-Hindi Parallel Corpus
"""

import os
import sys
import argparse
import yaml
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.transformer.model import TransformerSeq2Seq
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.data.dataset import TranslationDataset, pad_collate_fn
from ml.training.trainer import Trainer
from ml.inference.translator import Translator


def load_parallel_lines(src_file: str, tgt_file: str, max_samples: int = None):
    """Loads source and target text lines from parallel corpus files."""
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        raise FileNotFoundError(f"Missing corpus files: {src_file} or {tgt_file}. Run prepare_data.py first.")

    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        src_lines = [line.strip() for line in f_src if line.strip()]
        tgt_lines = [line.strip() for line in f_tgt if line.strip()]

    min_len = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:min_len]
    tgt_lines = tgt_lines[:min_len]

    if max_samples and max_samples > 0 and len(src_lines) > max_samples:
        src_lines = src_lines[:max_samples]
        tgt_lines = tgt_lines[:max_samples]

    return src_lines, tgt_lines


def main():
    parser = argparse.ArgumentParser(description="Train custom Transformer Seq2Seq model for English <-> Hindi NMT.")
    parser.add_argument("--config", type=str, default="ml/configs/baseline_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Override number of training sentence pairs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # 1. Load Configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else cfg["data"].get("max_train_samples")
    lr = args.lr or cfg["training"]["learning_rate"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    checkpoint_name = cfg["training"].get("checkpoint_name", "best_transformer_en_hi.pt")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("      TRAINING CUSTOM TRANSFORMER NMT (Attention Is All You Need)     ")
    print("=" * 70)
    print(f"[*] Configuration:   {args.config}")
    print(f"[*] Hardware Device: {device.upper()}")
    print(f"[*] Epochs:          {epochs}")
    print(f"[*] Batch Size:      {batch_size}")
    print(f"[*] Learning Rate:   {lr}")
    print(f"[*] Target Checkpt:  {checkpoint_path}")

    # 2. Load Tokenizers
    tok_en_path = "ml/tokenization/spm_en.model"
    tok_hi_path = "ml/tokenization/spm_hi.model"
    if not os.path.exists(tok_en_path) or not os.path.exists(tok_hi_path):
        raise FileNotFoundError(
            f"Trained SentencePiece tokenizers not found at {tok_en_path} or {tok_hi_path}. Run prepare_data.py first!"
        )

    src_tokenizer = SubwordTokenizer(model_path=tok_en_path)
    tgt_tokenizer = SubwordTokenizer(model_path=tok_hi_path)
    print(f"[+] Loaded Tokenizers | EN Vocab: {src_tokenizer.vocab_size:,} | HI Vocab: {tgt_tokenizer.vocab_size:,}")

    # 3. Load Processed Datasets
    processed_dir = cfg["data"]["processed_dir"]
    train_en_path = os.path.join(processed_dir, "train_en.txt")
    train_hi_path = os.path.join(processed_dir, "train_hi.txt")
    dev_en_path = os.path.join(processed_dir, "dev_en.txt")
    dev_hi_path = os.path.join(processed_dir, "dev_hi.txt")

    print("[*] Loading training corpus...")
    train_en, train_hi = load_parallel_lines(train_en_path, train_hi_path, max_samples=max_train_samples)
    print(f"[+] Loaded {len(train_en):,} training pairs.")

    print("[*] Loading validation benchmark (official dev split)...")
    dev_en, dev_hi = load_parallel_lines(dev_en_path, dev_hi_path, max_samples=None)
    print(f"[+] Loaded {len(dev_en):,} validation pairs.")

    train_dataset = TranslationDataset(train_en, train_hi, src_tokenizer, tgt_tokenizer)
    val_dataset = TranslationDataset(dev_en, dev_hi, src_tokenizer, tgt_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0,
    )

    # 4. Instantiate Custom Transformer Architecture
    model = TransformerSeq2Seq(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_encoder_layers=cfg["model"]["n_encoder_layers"],
        n_decoder_layers=cfg["model"]["n_decoder_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        pad_idx=SubwordTokenizer.PAD_ID,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Transformer Model Instantiated | Trainable Parameters: {total_params:,}")

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # 6. Training Loop
    print("=" * 70)
    print(f"[*] Commencing Training Loop for {epochs} Epoch(s)...")
    best_val_loss = float("inf")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate()
        epoch_duration = time.time() - epoch_start

        print(
            f"[*] Epoch {epoch:02d}/{epochs:02d} Complete | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                val_loss=val_loss,
                extra_meta={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "src_vocab_size": src_tokenizer.vocab_size,
                    "tgt_vocab_size": tgt_tokenizer.vocab_size,
                    "d_model": cfg["model"]["d_model"],
                    "n_heads": cfg["model"]["n_heads"],
                    "n_encoder_layers": cfg["model"]["n_encoder_layers"],
                    "n_decoder_layers": cfg["model"]["n_decoder_layers"],
                    "d_ff": cfg["model"]["d_ff"],
                    "total_parameters": total_params,
                    "train_samples": len(train_en),
                },
            )

    total_time = time.time() - start_total_time
    print("=" * 70)
    print(f"[+] Training Completed in {total_time:.1f}s | Best Val Loss: {best_val_loss:.4f}")
    print(f"[+] Saved Active Checkpoint: {checkpoint_path}")

    # 7. Post-Training Inference Verification
    print("=" * 70)
    print("[*] Performing Checkpoint Inference Verification...")
    test_translator = Translator(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        default_strategy="greedy",
    )
    sample_text = "Machine learning enables computers to learn from data."
    pred = test_translator.translate(sample_text, strategy="greedy")
    print(f"  Source Input:    {sample_text}")
    print(f"  Model Output:    {pred['translated_text']}")
    print(f"  Inference Speed: {pred['latency_ms']} ms")
    print("=" * 70)
    print("[+] Step 2 (Baseline Model Training & Checkpoint Generation) COMPLETE!")


if __name__ == "__main__":
    main()
