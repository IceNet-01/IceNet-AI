"""
Training pipeline optimized for Apple Silicon M4 Pro
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import logging
from tqdm import tqdm
import time

from icenet.core.engine import IceNetEngine
from icenet.core.config import Config
from icenet.training.callbacks import Callback, CheckpointCallback, LoggingCallback


logger = logging.getLogger(__name__)


class Trainer:
    """
    Training pipeline for IceNet models

    Optimized for Apple Silicon with MPS support
    """

    def __init__(
        self,
        engine: IceNetEngine,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Initialize trainer

        Args:
            engine: IceNet engine with model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            callbacks: List of callbacks (optional)
        """
        self.engine = engine
        self.config = engine.config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = engine.device

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision
        self.use_amp = self.config.training.mixed_precision in ["fp16", "bf16"]
        if self.use_amp and self.device.type == "mps":
            # MPS doesn't support GradScaler yet, use manual scaling
            self.scaler = None
        else:
            self.scaler = GradScaler() if self.use_amp else None

        # Setup callbacks
        self.callbacks = callbacks or []
        if not any(isinstance(cb, CheckpointCallback) for cb in self.callbacks):
            self.callbacks.append(CheckpointCallback(self.config.checkpoint))
        if not any(isinstance(cb, LoggingCallback) for cb in self.callbacks):
            self.callbacks.append(LoggingCallback(self.config.logging))

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")

        # Metrics
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        config = self.config.training

        # Get model parameters
        params = self.engine.model.parameters()

        if config.optimizer == "adam":
            optimizer = optim.Adam(
                params, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        elif config.optimizer == "adamw":
            optimizer = optim.AdamW(
                params, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        logger.info(f"Optimizer: {config.optimizer}")
        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        config = self.config.training

        total_steps = len(self.train_loader) * config.epochs

        if config.scheduler == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps,
            )
        elif config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=0
            )
        elif config.scheduler == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")

        logger.info(f"Scheduler: {config.scheduler}")
        return scheduler

    def train(self):
        """Run training loop"""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.training.epochs}")
        logger.info(f"Steps per epoch: {len(self.train_loader)}")

        # Call on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        try:
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch

                # Call on_epoch_begin callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)

                # Training epoch
                train_metrics = self._train_epoch()

                # Validation epoch
                val_metrics = {}
                if self.val_loader is not None:
                    val_metrics = self._validate_epoch()

                # Call on_epoch_end callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, train_metrics, val_metrics)

                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        finally:
            # Call on_train_end callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)

        logger.info("Training completed!")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.engine.train_mode()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Call on_batch_begin callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Training step
            loss = self._training_step(batch)
            total_loss += loss

            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

            # Call on_batch_end callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, {"loss": loss})

        avg_loss = total_loss / num_batches

        return {"loss": avg_loss}

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        if self.use_amp and self.scaler is not None:
            with autocast(device_type="cuda"):
                outputs = self.engine(batch["input_ids"])
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
        else:
            outputs = self.engine(batch["input_ids"])
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.engine.model.parameters(),
                    self.config.training.gradient_clip,
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.engine.model.parameters(),
                    self.config.training.gradient_clip,
                )

            # Optimizer step
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.engine.eval_mode()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.val_loader, desc="Validation")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = self.engine(batch["input_ids"])
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        return {"loss": avg_loss}

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log metrics"""
        log_str = f"Epoch {epoch + 1}/{self.config.training.epochs}"

        if train_metrics:
            log_str += " | Train:"
            for key, value in train_metrics.items():
                log_str += f" {key}={value:.4f}"

        if val_metrics:
            log_str += " | Val:"
            for key, value in val_metrics.items():
                log_str += f" {key}={value:.4f}"

        logger.info(log_str)

    def save_checkpoint(self, path: Path, metadata: Optional[Dict[str, Any]] = None):
        """Save training checkpoint"""
        self.engine.save_checkpoint(
            path=path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            metadata=metadata,
        )

    def load_checkpoint(self, path: Path):
        """Load training checkpoint"""
        metadata = self.engine.load_checkpoint(
            path=path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.current_epoch = metadata.get("epoch", 0)
        self.global_step = metadata.get("global_step", 0)

        return metadata
