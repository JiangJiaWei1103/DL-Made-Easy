"""
Cross-validator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building cv iterator for training
and evaluation processes.
"""
from argparse import Namespace
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .ts import GroupTimeSeriesSplit as GPTSSplit


def build_cv_iter(
    df: pd.DataFrame, args: Namespace
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Build and return the cv iterator.

    Parameters:
        df: raw data
        args: arguments driving training and evaluation processes

    Return:
        cv_iter: cross-validator
    """
    cv_scheme = args.cv_scheme
    n_folds = args.n_folds
    oof_size = args.oof_size
    group = args.group

    if cv_scheme == "gp":
        kf = GroupKFold(n_splits=n_folds)
        cv_iter = kf.split(X=df, groups=df[group])
    elif cv_scheme == "stratified":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_iter = kf.split(df, df["investment_id"])
    elif cv_scheme == "gpts":
        kf = GPTSSplit(n_folds=n_folds, oof_size=oof_size, groups=df[group])
        cv_iter = kf.split(df)

    return cv_iter
