"""
Custom trainer definitions for different training processes.
Author: JiaWei Jiang

This file contains diversified trainers, whose training logics are
inherited from `BaseTrainer`.
"""
import gc
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from evaluating.evaluator import Evaluator
from utils.early_stopping import EarlyStopping


class MainTrainer(BaseTrainer):
    """Main trainer.

    It's better to define different trainers for different models if
    there's a significant difference within training and evaluation
    processes (e.g., model input, advanced data processing, graph node
    sampling, customized multitask criterion definition).

    Parameters:
        proc_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        es: early stopping tracker
        train_loader: training data loader
        eval_loader: validation data loader
        scaler: scaling object
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
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        scaler: Optional[object] = None,
    ):
        super(MainTrainer, self).__init__(
            proc_cfg, model, loss_fn, optimizer, lr_skd, es, evaluator
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            tid, inv_id, x, y = self._process_batch_data(batch_data)

            # Forward pass
            output = self.model(x, inv_id)

            # Backpropagation
            if hasattr(self.loss_fn, "time_aware"):
                loss = self.loss_fn(output, torch.unsqueeze(y, dim=1), tid)
            else:
                loss = self.loss_fn(output, torch.unsqueeze(y, dim=1))
            loss.backward()
            self.optimizer.step()

            train_loss_total += loss.item()

            # Free mem.
            del tid, inv_id, x, y, output
            _ = gc.collect()

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run evaluation process for one epoch.

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
        """
        eval_loss_total = 0
        tids = None
        y_true = None
        y_pred = None

        self.model.eval()
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            tid, inv_id, x, y = self._process_batch_data(batch_data)

            # Forward pass
            output = self.model(x, inv_id)

            # Derive loss
            if hasattr(self.loss_fn, "time_aware"):
                loss = self.loss_fn(output, torch.unsqueeze(y, dim=1), tid)
            else:
                loss = self.loss_fn(output, torch.unsqueeze(y, dim=1))
            eval_loss_total += loss.item()

            # Record batched output
            if i == 0:
                tids = torch.squeeze(tid).detach().cpu()
                y_true = torch.squeeze(y).detach().cpu()
                y_pred = torch.squeeze(output).detach().cpu()
            else:
                tids = torch.cat((tids, torch.squeeze(tid).detach().cpu()))
                y_true = torch.cat((y_true, torch.squeeze(y).detach().cpu()))
                y_pred = torch.cat((y_pred, torch.squeeze(output).detach().cpu()))

            del tid, inv_id, x, y, output
            _ = gc.collect()

        eval_loss_avg = eval_loss_total / len(self.eval_loader)
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler, tids)

        return eval_loss_avg, eval_result

    def _process_batch_data(
        self, batch_data: Dict[str, Optional[Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Process and return batched raw data to facilitate further
        operations.

        Parameters:
            batch_data: batched raw data

        Return:
            tid: time identifiers
            inv_id: investment identifiers
            x: feature matrix
            y: groundtruths
        """
        # Retrieve batched raw data
        tid = batch_data["tid"].to(self.device)
        inv_id = batch_data["inv_id"].to(self.device)
        x = batch_data["x"].to(self.device)
        y = batch_data["y"].to(self.device)

        return tid, inv_id, x, y

    def _eval_with_best(self, best_model: Module) -> Dict[str, Dict[str, float]]:
        """Run final evaluation process with the best checkpoint.

        Parameters:
            best_model: model with the best evaluation loss or prf

        Return:
            final_prf_report: performance report of final evaluation
        """
        final_prf_report = {}
        self.model = best_model

        val_loader = self.eval_loader
        for datatype, dataloader in {
            "train": self.train_loader,
            "val": val_loader,
        }.items():
            self.eval_loader = dataloader
            eval_loss, eval_result = self._eval_epoch()
            final_prf_report[datatype] = eval_result

        return final_prf_report
