"""
Base class definition for all customized trainers.
Author: JiaWei Jiang
"""
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

import wandb
from evaluating.evaluator import Evaluator
from paths import DUMP_PATH
from utils.early_stopping import EarlyStopping


class BaseTrainer:
    """Base class for all customized trainers.

    Parameters:
        proc_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        es: early stopping tracker
        evaluator: task-specific evaluator
    """

    def __init__(
        self,
        proc_cfg: Dict[str, Any],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        es: EarlyStopping,
        evaluator: Evaluator,
    ):
        self.proc_cfg = proc_cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.es = es
        self.evaluator = evaluator

        self.device = proc_cfg["device"]
        self.epochs = proc_cfg["epochs"]

    def train_eval(self, proc_id: int) -> None:
        """Run train and evaluation processes for either one fold or
        one random seed (commonly used when training on whole dataset).

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.

        Return:
            None
        """
        val_loss_best = 1e18  # Monitored objective can be altered
        best_model = deepcopy(self.model)
        best_epoch = 0

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_result = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result
            val_metric_msg = ""
            for metric, score in val_result.items():
                val_metric_msg += f"{metric.upper()} {round(score, 4)} | "
            print(
                f"Epoch{epoch} | Training loss {train_loss} | "
                f"Validation loss {val_loss} | {val_metric_msg}"
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            # Record the best checkpoint
            if val_loss < val_loss_best:
                print(f"Validation performance improves at epoch {epoch}!!")
                val_loss_best = val_loss
                self._save_ckpt(proc_id, save_best_only=True)
                best_model = deepcopy(self.model)
                best_epoch = epoch

            # Check early stopping is triggered or not
            if self.es is not None:
                self.es.step(val_loss)
                if self.es.stop:
                    print(
                        f"Early stopping is triggered at epoch {epoch}, "
                        f"training process is halted."
                    )
                    break

        wandb.log({"best_epoch": best_epoch})

        # Run final evaluation
        final_prf_report = self._eval_with_best(best_model)
        wandb.log(final_prf_report)
        wandb.finish()

    @abstractmethod
    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run evaluation process for one epoch.

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_with_best(self, best_model: Module) -> Dict[str, Dict[str, float]]:
        """Run final evaluation process with the best checkpoint.

        Parameters:
            best_model: model with the best evaluation loss or prf

        Return:
            final_prf_report: performance report of final evaluation
        """
        raise NotImplementedError

    def _save_ckpt(self, proc_id: int, save_best_only: bool = True) -> None:
        """Save checkpoints.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.
            save_best_only: only checkpoint of the best epoch is saved

        Return:
            None
        """
        torch.save(
            self.model.state_dict(),
            os.path.join(DUMP_PATH, f"models/fold{proc_id}.pth"),
        )


#     def _resume_ckpt(self):
#         """Resume halted training and evaluation processes."""
#         pass
