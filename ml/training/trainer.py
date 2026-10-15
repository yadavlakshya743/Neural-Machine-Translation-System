"""Training orchestration, loss functions, and checkpoint management."""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.transformer.model import TransformerSeq2Seq
from ml.tokenization.tokenizer import SubwordTokenizer


from tqdm import tqdm


class LabelSmoothingLoss(nn.Module):
    """Cross Entropy loss with label smoothing to prevent overconfident predictions."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # x: (batch_size * seq_len, vocab_size)
        # target: (batch_size * seq_len)
        log_prob = torch.log_softmax(x, dim=-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(log_prob, true_dist)


class Trainer:
    """Manages model training, evaluation, validation loss, and checkpoint persistence."""

    def __init__(
        self,
        model: TransformerSeq2Seq,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 0.0005,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "models",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=SubwordTokenizer.PAD_ID, label_smoothing=0.1
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", unit="batch")

        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # Teacher forcing: input to decoder is tgt[:, :-1], expected target is tgt[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(src, tgt_input)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1)
            )
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            avg_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_so_far:.4f}"})

        avg_loss = total_loss / max(1, len(self.train_loader))
        return avg_loss

    def validate(self) -> float:
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc="           [Val]  ", unit="batch")

        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(pbar):
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1)
                )
                total_loss += loss.item()
                avg_so_far = total_loss / (batch_idx + 1)
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "avg_val": f"{avg_so_far:.4f}"})

        return total_loss / max(1, len(self.val_loader))

    def save_checkpoint(self, path: str, epoch: int, val_loss: float, extra_meta: Optional[Dict[str, Any]] = None):
        """Saves checkpoint with weights, optimizer state, and metadata."""
        meta = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if extra_meta:
            meta.update(extra_meta)
        torch.save(meta, path)
        print(f"[+] Best checkpoint saved: {path} (Val Loss: {val_loss:.4f})")

