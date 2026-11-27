"""Training orchestration: optimization schedule, mixed precision, checkpoint/resume, metrics."""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.dataset import TokenBucketBatchSampler
from ml.tokenization.tokenizer import SubwordTokenizer
from ml.transformer.model import TransformerSeq2Seq


@dataclass
class TrainConfig:
    learning_rate: float = 7e-4      # peak LR reached at the end of warmup
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 0.0001
    accum_steps: int = 1             # micro-batches per optimizer step
    use_amp: bool = True             # float16 autocast on CUDA
    epochs: int = 10
    patience: int = 3                # epochs without dev improvement before stopping (0 = off)
    select_metric: str = "bleu"      # "bleu" (higher is better) or "loss"
    save_every_steps: int = 1000     # periodic resumable checkpoint (optimizer steps)
    log_interval: int = 100          # optimizer steps between log lines
    max_hours: Optional[float] = None  # stop cleanly (and save) after this wall-clock budget


def inverse_sqrt_lr(step: int, warmup_steps: int) -> float:
    """LR multiplier: linear warmup to 1.0, then ~1/sqrt(step) decay.

    Same shape as Vaswani et al. (2017) eq. 3, but parameterised by the peak LR so the
    config value is directly interpretable.
    """
    step = max(step, 1)
    warmup_steps = max(warmup_steps, 1)
    return min(step / warmup_steps, math.sqrt(warmup_steps / step))


@dataclass
class TrainerState:
    epoch: int = 1                   # epoch currently being trained (1-based)
    batch_in_epoch: int = 0          # micro-batches of `epoch` already consumed
    global_step: int = 0             # optimizer steps
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_without_improvement: int = 0
    elapsed_sec: float = 0.0         # summed over all sessions
    status: str = "CONFIGURED"
    history: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {"train": [], "epochs": []})


class Trainer:
    """Manages model training, validation, checkpoint persistence and resumption."""

    def __init__(
        self,
        model: TransformerSeq2Seq,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        run_dir: str = "models/run",
        meta: Optional[Dict[str, Any]] = None,
        eval_fn: Optional[Callable[[TransformerSeq2Seq], Dict[str, float]]] = None,
    ):
        self.cfg = config or TrainConfig()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir
        self.meta = meta or {}
        self.eval_fn = eval_fn
        self.state = TrainerState()

        # No weight decay on biases and LayerNorm parameters.
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.dim() < 2 else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: inverse_sqrt_lr(s, self.cfg.warmup_steps)
        )
        self.amp = bool(self.cfg.use_amp and str(device).startswith("cuda"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=SubwordTokenizer.PAD_ID, label_smoothing=self.cfg.label_smoothing
        )
        os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------ paths
    @property
    def last_path(self) -> str:
        return os.path.join(self.run_dir, "last.pt")

    @property
    def best_path(self) -> str:
        return os.path.join(self.run_dir, "best.pt")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.run_dir, "metrics.json")

    # ------------------------------------------------------------- core steps
    def _loss(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Teacher forcing: decoder sees tgt[:, :-1] and predicts tgt[:, 1:].
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.model(src, tgt[:, :-1])
        return self.criterion(logits.float().reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

    def validate(self) -> float:
        if not self.val_loader:
            return float("nan")
        self.model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                n = int((tgt[:, 1:] != SubwordTokenizer.PAD_ID).sum())
                total += self._loss(src, tgt).item() * n
                tokens += n
        return total / max(1, tokens)

    # ------------------------------------------------------------ persistence
    def _payload(self, full: bool) -> Dict[str, Any]:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "meta": {**self.meta, "model_config": self.model.config},
            "trainer_state": asdict(self.state),
            "train_config": asdict(self.cfg),
        }
        if full:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
            payload["scaler_state_dict"] = self.scaler.state_dict()
        return payload

    def _atomic_save(self, payload: Dict[str, Any], path: str) -> None:
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)  # never leaves a half-written checkpoint behind

    def save_last(self) -> None:
        self._atomic_save(self._payload(full=True), self.last_path)

    def save_best(self) -> None:
        # Inference checkpoint: weights + metadata only (much smaller than last.pt).
        self._atomic_save(self._payload(full=False), self.best_path)

    def write_metrics(self) -> None:
        tmp = self.metrics_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"meta": self.meta, "train_config": asdict(self.cfg), "state": asdict(self.state)},
                f, indent=2, ensure_ascii=False, default=str,
            )
        os.replace(tmp, self.metrics_path)

    def resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        old_meta = ckpt.get("meta", {})
        for side in ("src", "tgt"):
            old = old_meta.get("tokenizers", {}).get(side, {}).get("sha256")
            new = self.meta.get("tokenizers", {}).get(side, {}).get("sha256")
            if old and new and old != new:
                raise RuntimeError(
                    f"Cannot resume: {side} tokenizer differs from the one used for {path}. "
                    "Restore the original tokenizer files instead of retraining them."
                )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        st = ckpt["trainer_state"]
        self.state = TrainerState(**st)
        print(
            f"[+] Resumed from {path}: epoch {self.state.epoch}, batch {self.state.batch_in_epoch}, "
            f"step {self.state.global_step}, best {self.cfg.select_metric}={self.state.best_score}"
        )

    # -------------------------------------------------------------- main loop
    def _is_better(self, score: float) -> bool:
        if self.state.best_score is None or math.isnan(self.state.best_score):
            return not math.isnan(score)
        if self.cfg.select_metric == "loss":
            return score < self.state.best_score
        return score > self.state.best_score

    def fit(self) -> str:
        """Trains until epochs are done, early stopping triggers, or the time budget runs out.

        Returns the final status: COMPLETED, EARLY_STOPPED or TIME_BUDGET_REACHED.
        """
        cfg = self.cfg
        session_start = time.time()
        elapsed_before = self.state.elapsed_sec
        sampler = getattr(self.train_loader, "batch_sampler", None)
        self.state.status = "TRAINING"

        def out_of_time() -> bool:
            return cfg.max_hours is not None and (time.time() - session_start) > cfg.max_hours * 3600

        def checkpoint_now():
            self.state.elapsed_sec = elapsed_before + (time.time() - session_start)
            self.save_last()
            self.write_metrics()

        while self.state.epoch <= cfg.epochs:
            epoch = self.state.epoch
            if isinstance(sampler, TokenBucketBatchSampler):
                sampler.set_epoch(epoch, start_batch=self.state.batch_in_epoch)
            elif self.state.batch_in_epoch:
                print("[!] Sampler cannot skip batches; restarting the epoch from its beginning.")
                self.state.batch_in_epoch = 0

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            win_loss, win_tokens, win_start, micro = 0.0, 0, time.time(), 0

            for src, tgt in self.train_loader:
                src, tgt = src.to(self.device, non_blocking=True), tgt.to(self.device, non_blocking=True)
                loss = self._loss(src, tgt)
                self.scaler.scale(loss / cfg.accum_steps).backward()

                n_tok = int((tgt[:, 1:] != SubwordTokenizer.PAD_ID).sum())
                win_loss += loss.item() * n_tok
                win_tokens += n_tok
                micro += 1
                self.state.batch_in_epoch += 1

                if micro % cfg.accum_steps != 0:
                    continue

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.state.global_step += 1

                if self.state.global_step % cfg.log_interval == 0:
                    dt = max(1e-6, time.time() - win_start)
                    entry = {
                        "step": self.state.global_step,
                        "epoch": epoch,
                        "loss": round(win_loss / max(1, win_tokens), 4),
                        "lr": self.scheduler.get_last_lr()[0],
                        "tgt_tokens_per_sec": round(win_tokens / dt),
                    }
                    self.state.history["train"].append(entry)
                    print(
                        f"  step {entry['step']:>7} | epoch {epoch} | loss {entry['loss']:.4f} | "
                        f"lr {entry['lr']:.2e} | {entry['tgt_tokens_per_sec']:,} tgt tok/s",
                        flush=True,
                    )
                    win_loss, win_tokens, win_start = 0.0, 0, time.time()

                if self.state.global_step % cfg.save_every_steps == 0:
                    checkpoint_now()

                if out_of_time():
                    self.state.status = "TIME_BUDGET_REACHED"
                    checkpoint_now()
                    print(f"[!] Time budget of {cfg.max_hours}h reached; saved {self.last_path}. "
                          "Resume with --resume to continue.")
                    return self.state.status

            # ---- end of epoch: validate, select, early-stop
            val_loss = self.validate()
            record: Dict[str, Any] = {"epoch": epoch, "step": self.state.global_step,
                                      "val_loss": round(val_loss, 4)}
            if self.eval_fn is not None:
                record.update(self.eval_fn(self.model))
                self.model.train()
            score = record.get("dev_bleu") if cfg.select_metric == "bleu" else val_loss
            if score is None:
                score = val_loss
            improved = self._is_better(score)
            record["best"] = improved
            self.state.history["epochs"].append(record)
            print(f"[*] Epoch {epoch} done | " + " | ".join(f"{k} {v}" for k, v in record.items()
                                                           if k not in ("epoch",)), flush=True)

            if improved:
                self.state.best_score, self.state.best_epoch = score, epoch
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1

            self.state.epoch += 1
            self.state.batch_in_epoch = 0
            if improved:
                self.save_best()
                print(f"[+] New best checkpoint: {self.best_path}")
            checkpoint_now()

            if cfg.patience and self.state.epochs_without_improvement >= cfg.patience:
                self.state.status = "EARLY_STOPPED"
                checkpoint_now()
                print(f"[!] Early stopping: no improvement for {cfg.patience} epochs.")
                return self.state.status

        self.state.status = "COMPLETED"
        checkpoint_now()
        return self.state.status
