"""Training pipeline for TDE spiking object detector.

Implements:
- Config-driven training with TOML configs
- SGD optimizer with cosine annealing + warmup
- Checkpoint management (top-2 by val_loss)
- Early stopping
- TensorBoard + JSON logging
- Resume from checkpoint

Reference: arXiv 2512.02447, arXiv 2407.20708 (SpikeYOLO training)
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anima_tde.dataset import build_dataset, collate_fn
from anima_tde.losses import build_loss
from anima_tde.model import build_model
from anima_tde.utils import set_seed


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup, resume-aware."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]


class CheckpointManager:
    """Manages top-K checkpoints by validation metric."""

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 2,
        metric: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict,
        metric_value: float,
        step: int,
    ) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort: best first
        self.history.sort(
            key=lambda x: x[0], reverse=(self.mode == "max")
        )

        # Keep top K
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        # Save best
        best_path = self.save_dir / "best.pth"
        if self.history:
            best_src = self.history[0][1]
            shutil.copy2(best_src, best_path)

        return path


class EarlyStopping:
    """Early stopping monitor."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train(config: dict, resume: str | None = None) -> None:
    """Run training loop.

    Args:
        config: Parsed TOML configuration.
        resume: Path to checkpoint to resume from.
    """
    train_cfg = config["training"]
    ckpt_cfg = config["checkpoint"]
    log_cfg = config["logging"]
    es_cfg = config.get("early_stopping", {})

    # Seed
    set_seed(train_cfg.get("seed", 42))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Model
    model = build_model(config).to(device)
    param_count = model.count_parameters()
    print(f"[MODEL] {param_count / 1e6:.2f}M parameters")
    print(f"[MODEL] TDE variant: {config['model'].get('tde_variant', 'sda')}")

    # Loss
    criterion = build_loss(config)

    # Optimizer
    lr = train_cfg["learning_rate"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
    )

    # Datasets
    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")

    batch_size = train_cfg.get("batch_size", 16)
    if batch_size == "auto":
        batch_size = 16  # Placeholder -- use /gpu-batch-finder

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    # Scheduler
    epochs = train_cfg["epochs"]
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=train_cfg.get("min_lr", 1e-6),
    )

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=ckpt_cfg["output_dir"],
        keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        metric=ckpt_cfg.get("metric", "val_loss"),
        mode=ckpt_cfg.get("mode", "min"),
    )

    # Early stopping
    early_stop = None
    if es_cfg.get("enabled", True):
        early_stop = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 1e-4),
            mode=ckpt_cfg.get("mode", "min"),
        )

    # Logging
    log_dir = Path(log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / "training_history.jsonl"

    tb_dir = Path(log_cfg["tensorboard_dir"])
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(str(tb_dir))
    except ImportError:
        print("[WARN] TensorBoard not available, skipping TB logging")

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        print(f"[RESUME] Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] Starting from epoch {start_epoch}, step {global_step}")

    # Mixed precision
    use_amp = train_cfg.get("precision", "bf16") != "fp32" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if train_cfg.get("precision") == "bf16" else torch.float16

    # Print config summary
    print(f"[CONFIG] epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"[CONFIG] train_samples={len(train_dataset)}, val_samples={len(val_dataset)}")
    print(f"[CONFIG] warmup_steps={warmup_steps}, total_steps={total_steps}")
    print("[TRAIN] Starting training...")

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for _batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                predictions = model(images)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict["total"]

            # NaN check
            if torch.isnan(loss):
                print(f"[FATAL] Loss is NaN at epoch {epoch}, step {global_step}")
                print("[FIX] Reduce lr, check data, check gradient clipping")
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Step logging
            if global_step % 50 == 0:
                lr_now = scheduler.get_lr()
                print(
                    f"  [Step {global_step}] loss={loss.item():.4f} "
                    f"box={loss_dict['box'].item():.4f} "
                    f"obj={loss_dict['obj'].item():.4f} "
                    f"cls={loss_dict['cls'].item():.4f} "
                    f"lr={lr_now:.6f}"
                )

            # Step-based checkpointing
            save_every = ckpt_cfg.get("save_every_n_steps", 500)
            if global_step % save_every == 0:
                # Quick val for checkpoint metric
                val_loss = _validate(model, val_loader, criterion, device, use_amp, amp_dtype)
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "val_loss": val_loss,
                    "config": config,
                }
                ckpt_manager.save(state, val_loss, global_step)
                model.train()

        # Epoch end
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = _validate(model, val_loader, criterion, device, use_amp, amp_dtype)
        elapsed = time.time() - t0
        lr_now = scheduler.get_lr()

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"lr={lr_now:.6f} "
            f"time={elapsed:.1f}s"
        )

        # Log
        metrics = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            "time_s": elapsed,
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if tb_writer:
            tb_writer.add_scalar("loss/train", avg_train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("lr", lr_now, epoch)

        # Checkpoint
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "val_loss": val_loss,
            "config": config,
        }
        ckpt_manager.save(state, val_loss, global_step)

        # Early stopping
        if early_stop and early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs.")
            break

    print("[TRAIN] Training complete.")
    if tb_writer:
        tb_writer.close()


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            predictions = model(images)
            loss_dict = criterion(predictions, targets)

        total_loss += loss_dict["total"].item()
        count += 1

    return total_loss / max(count, 1)
