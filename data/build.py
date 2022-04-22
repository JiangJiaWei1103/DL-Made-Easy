"""
Dataloader building logic.
Author: JiaWei Jiang

This file contains the basic logic of building dataloaders for training
and evaluation processes.
"""
from typing import Any, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from .dataset import MLPDataset


def build_dataloaders(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    **dataset_cfg: Any,  # Temporary workaround
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create and return train and validation data loaders.

    Parameters:
        df_tr: training data
        df_val: validation data
        model_name: name of model architecture
        batch_size: number of samples per batch
        shuffle: whether to shuffle samples every epoch
        num_workers: number of subprocesses used to load data
        dataset_cfg: hyperparameters of customized dataset

    Return:
        train_loader: training data loader
        val_loader: validation data loader
    """
    if model_name == "MLP":
        dataset = MLPDataset
        collate_fn = None

    train_loader = DataLoader(
        dataset(df_tr, **dataset_cfg),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if df_val is not None:
        val_loader = DataLoader(
            dataset(df_val, **dataset_cfg),
            batch_size=batch_size,
            shuffle=False,  # Hard-coded
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader
    else:
        return train_loader, None


# Customized collate wrapper definitions
# def _collate_wrapper(batch) -> None:
#     """Mini-batching wrapper for training model architecture with graph
#     structure learner.

#     For more detailed information, please see https://pytorch.org/docs/
#     stable/data.html#loading-batched-and-non-batched-data
#     """
#     pass
