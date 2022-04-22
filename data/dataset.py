"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.
"""
from typing import Any, Dict

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from metadata import FEAT_COLS


class MLPDataset(Dataset):
    """Dataset for naive fully-connected structure.

    Parameters:
        df: processed data
        dataset_cfg: hyperparameters of customized dataset
    """

    def __init__(self, df: pd.DataFrame, **dataset_cfg: Any):
        self.tid = df["time_id"]
        self.inv_id = df["investment_id"]
        self.feat_map = df[FEAT_COLS]
        self.y = df["target"]
        self.len = df.shape[0]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        tid = torch.tensor(self.tid.iloc[idx], dtype=torch.int16)
        inv_id = torch.tensor(self.inv_id.iloc[idx], dtype=torch.int32)
        feat_map = torch.tensor(self.feat_map.iloc[idx, :], dtype=torch.float32)
        y = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        return {"tid": tid, "inv_id": inv_id, "x": feat_map, "y": y}
