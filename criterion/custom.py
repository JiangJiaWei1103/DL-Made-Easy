"""
Custom loss criterion definitions.
Author: JiaWei Jiang

If users want to use customized loss criterion, the corresponding class
should be defined in this file.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MaxMarginLoss(_Loss):
    """Max-margin loss.

    For more detailed information, please refer to https://arxiv.org/
    pdf/1809.09441.pdf

    Parameters:
        time_aware: whether loss is time-dependent or not
        task_balance: balance between relational and absolute loss
            criteria
        loss_abs: name of the absolute loss criterion
    """

    def __init__(
        self, time_aware: bool = True, task_balance: float = 0.2, loss_abs: str = "l2"
    ):
        super(MaxMarginLoss, self).__init__()
        self.time_aware = time_aware
        self.task_balance = task_balance
        self.loss_abs = nn.MSELoss() if loss_abs == "l2" else nn.L1Loss()

    def forward(
        self, y_pred: Tensor, y_true: Tensor, tid: Optional[Tensor] = None
    ) -> Tensor:
        distance_loss = self.loss_abs(y_pred, y_true)
        if self.time_aware:
            _, split_sizes = torch.unique(tid, return_counts=True)
            y_pred = torch.split(y_pred, tuple(split_sizes))
            y_true = torch.split(y_true, tuple(split_sizes))

        max_margin_loss = torch.zeros(1)
        for y_pred_t, y_true_t in zip(y_pred, y_true):
            y_pred_t = y_pred_t.squeeze()
            y_true_t = y_true_t.squeeze()
            y_pred_t_std = torch.std(y_pred_t) + 1e-6
            y_true_t_std = torch.std(y_true_t) + 1e-6
            term_self = (y_true_t * y_pred_t).unsqueeze(1)  # Enable broadcast
            term_cross = torch.outer(y_true_t, y_pred_t)

            pairwise_loss = (term_cross + term_cross.T) - term_self - term_self.T
            pairwise_loss = F.relu(pairwise_loss)
            max_margin_loss += torch.mean(pairwise_loss)
        max_margin_loss = self.task_balance * max_margin_loss + distance_loss

        return max_margin_loss
