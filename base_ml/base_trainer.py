# -*- coding: utf-8 -*-
# Base Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import os
from abc import abstractmethod
from typing import Tuple, Union

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import wandb
from base_ml.base_early_stopping import EarlyStopping
from pathlib import Path
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from utils.efficiency_metrics import EfficiencyRecorder
from utils.tools import flatten_dict


class BaseTrainer:
    """
    Base class for all trainers with important ML components

    Args:
        model (nn.Module):  Model that should be trained
        loss_fn (_Loss): Loss function
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional): Early Stopping Class. Defaults to None.
        accum_iter (int, optional): Accumulation steps for gradient accumulation.
            Provide a number greater than 1 for activating gradient accumulation. Defaults to 1.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
        log_images (bool, optional): If images should be logged to WandB. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        accum_iter: int = 1,
        mixed_precision: bool = False,
        log_images: bool = False,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.logdir = Path(logdir)
        self.early_stopping = early_stopping
        self.accum_iter = accum_iter
        self.start_epoch = 0
        self.experiment_config = experiment_config
        self.log_images = log_images
        self.mixed_precision = mixed_precision
        self.checkpointing_config = experiment_config.get("checkpointing")
        self.last_validation_metrics = None
        self.best_validation_metrics = None
        self.efficiency_recorder = None
        if self.checkpointing_config is not None:
            defaults = {
                "save_best": True,
                "save_last": True,
                "keep_last_n": 1,
                "save_every": 1,
                "delete_intermediate_checkpoints": True,
            }
            defaults.update(self.checkpointing_config)
            self.checkpointing_config = defaults
            if int(defaults["save_every"]) < 1:
                raise ValueError("checkpointing.save_every must be >= 1")
            if int(defaults["keep_last_n"]) < 0:
                raise ValueError("checkpointing.keep_last_n must be >= 0")
            if defaults["save_last"] and int(defaults["keep_last_n"]) < 1:
                raise ValueError(
                    "checkpointing.keep_last_n must be >= 1 when save_last=true"
                )
            self.logger.info(f"Checkpointing policy: {self.checkpointing_config}")
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None
        
        # Initialize Tensorboard writer
        # make directory for tensorboard logging
        (self.logdir / "gradients_tracking").mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=self.logdir / "gradients_tracking")

    @abstractmethod
    def train_epoch(
        self, epoch: int, train_loader: DataLoader, **kwargs
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_loader (DataLoader): Train dataloader

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        raise NotImplementedError

    @abstractmethod
    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float]:
        """Training logic for an validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Tuple[dict, dict, float]: wandb logging dictionaries and early_stopping_metric
                * Scalar metrics
                * Image metrics
                * Early Stopping metric as float
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: object, batch_idx: int, num_batches: int):
        """Training logic for one training batch

        Args:
            batch (object): A training batch
            batch_idx (int): Current batch index
            num_batches (int): Maximum number of batches

        Raises:
            NotImplementedError: Needs to be implemented
        """

        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx: int):
        """Training logic for one validation batch

        Args:
            batch (object): A training batch
            batch_idx (int): Current batch index

        Raises:
            NotImplementedError: Needs to be implemented
        """

    def fit(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metric_init: dict = None,
        eval_every: int = 1,
        **kwargs,
    ):
        """Fitting function to start training and validation of the trainer

        Args:
            epochs (int): Number of epochs the network should be training
            train_dataloader (DataLoader): Dataloader with training data
            val_dataloader (DataLoader): Dataloader with validation data
            metric_init (dict, optional): Initialization dictionary with scalar metrics that should be initialized for startup.
                This is just import for logging with wandb if you want to have the plots properly scaled.
                The data in the the metric dictionary is used as values for epoch 0 (before training has startetd).
                If not provided, step 0 (epoch 0) is not logged. Should have the same scalar keys as training and validation epochs report.
                For more information, you should have a look into the train_epoch and val_epoch methods where the wandb logging dicts are assembled.
                Defaults to None.
            eval_every (int, optional): How often the network should be evaluated (after how many epochs). Defaults to 1.
            **kwargs
        """

        self.logger.info(f"Starting training, total number of epochs: {epochs}")
        if metric_init is not None and self.start_epoch == 0:
            wandb.log(metric_init, step=0)

        efficiency_config = self.experiment_config.get("efficiency", {})
        if bool(efficiency_config.get("enabled", False)):
            self.efficiency_recorder = EfficiencyRecorder(
                output_path=self.logdir / "efficiency_metrics.json",
                mode="training",
                experiment_config=self.experiment_config,
                model=self.model,
                device=self.device,
                batch_size=train_dataloader.batch_size,
            )
            # Model/device, optimizer, scheduler, datasets and loaders are initialized.
            # Start the measured interval immediately before the first epoch.
            self.efficiency_recorder.start()

        for epoch in range(self.start_epoch, epochs):
            should_stop = False
            if self.efficiency_recorder is not None:
                self.efficiency_recorder.start_epoch()
            # training epoch
            self.logger.info(f"Epoch: {epoch+1}/{epochs}")
            if self.experiment_config["adapters"].get("adapter_type", None) != "freeze":
                train_scalar_metrics, train_image_metrics = self.train_epoch(
                    epoch, train_dataloader, **kwargs
                )
                wandb.log(train_scalar_metrics, step=epoch + 1)
                if self.log_images:
                    wandb.log(train_image_metrics, step=epoch + 1)
            else:
                self.logger.info("Adapter training is frozen - no training epoch")
            if ((epoch + 1) % eval_every) == 0:
                # validation epoch
                (
                    val_scalar_metrics,
                    val_image_metrics,
                    early_stopping_metric,
                ) = self.validation_epoch(epoch, val_dataloader)
                wandb.log(val_scalar_metrics, step=epoch + 1)
                if self.log_images:
                    wandb.log(val_image_metrics, step=epoch + 1)

            # log learning rate
            curr_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "Learning-Rate/Learning-Rate": curr_lr,
                },
                step=epoch + 1,
            )
            if (epoch + 1) % eval_every == 0:
                # early stopping
                if self.early_stopping is not None:
                    best_model = self.early_stopping(early_stopping_metric, epoch)
                    if best_model:
                        self.best_validation_metrics = dict(val_scalar_metrics)
                        if self._checkpoint_option("save_best", True):
                            self.logger.info("New best model - save checkpoint")
                            self.save_checkpoint(epoch, "model_best.pth")
                        else:
                            self.logger.info(
                                "New best model - checkpointing.save_best=false, not saved"
                            )
                    elif self.early_stopping.early_stop:
                        self.logger.info("Performing early stopping!")
                        should_stop = True
                self.last_validation_metrics = dict(val_scalar_metrics)

            if self._should_save_epoch_checkpoint(
                epoch_number=epoch + 1,
                total_epochs=epochs,
                early_stopping=should_stop,
            ):
                checkpoint_path = self.save_checkpoint(
                    epoch, f"checkpoint_{epoch+1}.pth"
                )
                self.logger.info(f"Saved checkpoint for epoch {epoch+1}: {checkpoint_path}")
                self._prune_epoch_checkpoints()

            if should_stop:
                if self.efficiency_recorder is not None:
                    self.efficiency_recorder.finish_epoch()
                break

            # scheduling
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(float(val_scalar_metrics["Loss/Validation"]))
            else:
                self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.debug(f"Old lr: {curr_lr:.6f} - New lr: {new_lr:.6f}")
            if self.efficiency_recorder is not None:
                self.efficiency_recorder.finish_epoch()

        self._deduplicate_best_and_latest()
        self._log_retained_checkpoints()
        if self.efficiency_recorder is not None:
            self.efficiency_recorder.finish_interval(
                status="training_finished_pending_checkpoint"
            )

    def finalize_efficiency(self, checkpoint_path: Path = None) -> None:
        """Mark an instrumented run complete after its final checkpoint exists."""
        if self.efficiency_recorder is not None:
            self.efficiency_recorder.finalize_training(checkpoint_path)

    def _checkpoint_option(self, key: str, legacy_default):
        if self.checkpointing_config is None:
            return legacy_default
        return self.checkpointing_config[key]

    def _should_save_epoch_checkpoint(
        self, epoch_number: int, total_epochs: int, early_stopping: bool
    ) -> bool:
        if self.checkpointing_config is None:
            return True
        save_every = int(self.checkpointing_config["save_every"])
        scheduled = epoch_number % save_every == 0
        final_epoch = epoch_number == total_epochs or early_stopping
        return scheduled or (
            bool(self.checkpointing_config["save_last"]) and final_epoch
        )

    def _epoch_checkpoints(self):
        checkpoint_dir = self.logdir / "checkpoints"
        checkpoints = []
        for path in checkpoint_dir.glob("checkpoint_*.pth"):
            try:
                epoch = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            checkpoints.append((epoch, path))
        return sorted(checkpoints, key=lambda item: item[0])

    def _prune_epoch_checkpoints(self) -> None:
        if self.checkpointing_config is None:
            return
        if not self.checkpointing_config["delete_intermediate_checkpoints"]:
            return
        keep_last_n = int(self.checkpointing_config["keep_last_n"])
        checkpoints = self._epoch_checkpoints()
        to_delete = checkpoints[:-keep_last_n] if keep_last_n else checkpoints
        for _, path in to_delete:
            path.unlink()
            self.logger.info(f"Deleted intermediate checkpoint: {path}")
        kept = [str(path) for _, path in self._epoch_checkpoints()]
        self.logger.info(f"Retained epoch checkpoints: {kept}")

    def _log_retained_checkpoints(self) -> None:
        checkpoint_dir = self.logdir / "checkpoints"
        if not checkpoint_dir.exists():
            return
        kept = sorted(path.name for path in checkpoint_dir.glob("*.pth"))
        self.logger.info(f"Final retained checkpoints: {kept}")

    def _deduplicate_best_and_latest(self) -> None:
        if self.checkpointing_config is None or self.early_stopping is None:
            return
        if not self.checkpointing_config["save_best"]:
            return
        checkpoints = self._epoch_checkpoints()
        if not checkpoints or self.early_stopping.best_epoch is None:
            return
        latest_epoch, latest_path = checkpoints[-1]
        if latest_epoch != int(self.early_stopping.best_epoch) + 1:
            return
        best_path = self.logdir / "checkpoints" / "model_best.pth"
        if not best_path.is_file():
            return
        temporary = best_path.with_name(".model_best.pth.link-tmp")
        try:
            temporary.unlink(missing_ok=True)
            os.link(latest_path, temporary)
            os.replace(temporary, best_path)
            self.logger.info(
                f"Best and last are epoch {latest_epoch}; hard-linked "
                f"{best_path.name} to {latest_path.name} to avoid duplicate storage"
            )
        except OSError as error:
            temporary.unlink(missing_ok=True)
            self.logger.warning(
                f"Could not hard-link best and last checkpoints; keeping both files: {error}"
            )

    def save_checkpoint(self, epoch: int, checkpoint_name: str):
        if self.early_stopping is None:
            best_metric = None
            best_epoch = None
        else:
            best_metric = self.early_stopping.best_metric
            best_epoch = self.early_stopping.best_epoch

        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": flatten_dict(wandb.config),
            "wandb_id": wandb.run.id,
            "logdir": str(self.logdir.resolve()),
            "run_name": str(Path(self.logdir).name),
            "scaler_state_dict": self.scaler.state_dict()
            if self.scaler is not None
            else None,
        }

        checkpoint_dir = self.logdir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        filename = checkpoint_dir / checkpoint_name
        temporary = checkpoint_dir / f".{checkpoint_name}.tmp-{os.getpid()}"
        try:
            torch.save(state, temporary)
            os.replace(temporary, filename)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.logger.info(f"Checkpoint safely written: {filename}")
        return filename

    def resume_checkpoint(self, checkpoint):
        self.logger.info(f"Loading checkpoint")
        self.logger.info("Loading Model")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Loading Optimizer state dict")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.early_stopping is not None:
            self.early_stopping.best_metric = checkpoint["best_metric"]
            self.early_stopping.best_epoch = checkpoint["best_epoch"]
        if self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.info(f"Checkpoint epoch in reality in the checkpoint file: {int(checkpoint['epoch'])}")
        self.logger.info(f"Checkpoint epoch to print is: {int(checkpoint['epoch'])+1}")
        self.start_epoch = int(checkpoint["epoch"])+1
        self.logger.info(f"Next epoch to print is: {self.start_epoch + 1}")
