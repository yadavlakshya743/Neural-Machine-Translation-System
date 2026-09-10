#!/usr/bin/env python3
"""CLI training script for Transformer English <-> Hindi NMT."""

import argparse
import yaml
import torch
from ml.transformer.model import TransformerSeq2Seq
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.data.dataset import TranslationDataset, pad_collate_fn
from ml.training.trainer import Trainer
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train Transformer NMT model")
    parser.add_argument("--config", type=str, default="ml/configs/baseline_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[*] Loaded configuration from {args.config}")
    print(f"[*] Initializing model: {cfg['model']['name']}")

    # Initialize tokenizers
    src_tokenizer = SubwordTokenizer()
    tgt_tokenizer = SubwordTokenizer()

    # Instantiate model
    model = TransformerSeq2Seq(
        src_vocab_size=cfg["model"]["src_vocab_size"],
        tgt_vocab_size=cfg["model"]["tgt_vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_encoder_layers=cfg["model"]["n_encoder_layers"],
        n_decoder_layers=cfg["model"]["n_decoder_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
    )

    print(f"[+] Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Prepare dummy dataset for baseline demonstration if raw data not yet placed
    src_sample = ["The quick brown fox jumps over the lazy dog.", "Machine translation is fascinating."]
    tgt_sample = ["तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।", "मशीन अनुवाद आकर्षक है।"]

    train_dataset = TranslationDataset(src_sample, tgt_sample, src_tokenizer, tgt_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=cfg["training"]["learning_rate"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
    )

    print("[*] Running 1 warmup training epoch...")
    loss = trainer.train_epoch(epoch=1)
    print(f"[+] Warmup Epoch completed. Loss: {loss:.4f}")

    checkpoint_path = f"{cfg['training']['checkpoint_dir']}/baseline_en_hi.pt"
    trainer.save_checkpoint(checkpoint_path, epoch=1, val_loss=loss)


if __name__ == "__main__":
    main()
