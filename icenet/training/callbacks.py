"""Training callbacks for IceNet"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class"""

    def on_train_begin(self, trainer):
        """Called at the beginning of training"""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training"""
        pass

    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(
        self, trainer, epoch: int, train_metrics: Dict, val_metrics: Dict
    ):
        """Called at the end of each epoch"""
        pass

    def on_batch_begin(self, trainer, batch_idx: int):
        """Called at the beginning of each batch"""
        pass

    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict):
        """Called at the end of each batch"""
        pass


class CheckpointCallback(Callback):
    """Callback for saving checkpoints"""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_total_limit = config.save_total_limit
        self.save_best_only = config.save_best_only
        self.metric_for_best = config.metric_for_best

        self.best_metric = float("inf")
        self.checkpoints = []

    def on_epoch_end(
        self, trainer, epoch: int, train_metrics: Dict, val_metrics: Dict
    ):
        """Save checkpoint at end of epoch"""
        # Determine which metrics to use
        metrics = val_metrics if val_metrics else train_metrics

        if not metrics:
            return

        # Check if we should save
        current_metric = metrics.get(self.metric_for_best, float("inf"))

        should_save = False
        is_best = False

        if not self.save_best_only:
            should_save = True
        elif current_metric < self.best_metric:
            should_save = True
            is_best = True
            self.best_metric = current_metric

        if should_save:
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch{epoch + 1}_{timestamp}.pt"

            if is_best:
                filename = f"checkpoint_best_{timestamp}.pt"

            checkpoint_path = self.output_dir / filename

            # Save checkpoint
            metadata = {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_metric": self.best_metric,
            }

            trainer.save_checkpoint(checkpoint_path, metadata)

            # Track checkpoints
            self.checkpoints.append(checkpoint_path)

            # Remove old checkpoints if limit exceeded
            if len(self.checkpoints) > self.save_total_limit:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists() and not old_checkpoint.name.startswith("checkpoint_best"):
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")


class LoggingCallback(Callback):
    """Callback for logging training progress"""

    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = config.use_tensorboard
        self.use_wandb = config.use_wandb

        # Initialize tensorboard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir)
                logger.info(f"TensorBoard logging to {self.log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.use_tensorboard = False

        # Initialize wandb
        if self.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project or "icenet",
                    name=config.wandb_run_name,
                    dir=self.log_dir,
                )
                logger.info("W&B logging enabled")
            except ImportError:
                logger.warning("W&B not available")
                self.use_wandb = False

    def on_epoch_end(
        self, trainer, epoch: int, train_metrics: Dict, val_metrics: Dict
    ):
        """Log metrics at end of epoch"""
        # Log to tensorboard
        if self.use_tensorboard:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)

            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)

            # Log learning rate
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", lr, epoch)

        # Log to wandb
        if self.use_wandb:
            import wandb

            log_dict = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
                "lr": trainer.optimizer.param_groups[0]["lr"],
            }
            wandb.log(log_dict)

    def on_train_end(self, trainer):
        """Cleanup at end of training"""
        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            import wandb

            wandb.finish()


class EarlyStoppingCallback(Callback):
    """Callback for early stopping"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, metric: str = "loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric

        self.best_metric = float("inf")
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(
        self, trainer, epoch: int, train_metrics: Dict, val_metrics: Dict
    ):
        """Check if training should stop"""
        # Use validation metrics if available, otherwise training metrics
        metrics = val_metrics if val_metrics else train_metrics

        if not metrics or self.metric not in metrics:
            return

        current_metric = metrics[self.metric]

        # Check if metric improved
        if current_metric < self.best_metric - self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            self.should_stop = True
