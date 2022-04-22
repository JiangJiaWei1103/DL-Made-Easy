"""
Data processor definitions.
Author: JiaWei Jiang

This file contains the definition of data processor cleaning and
processing raw data before entering modeling phase. Because data
processing is case-specific, so I leave this part to users to customize
the procedure.
"""
import gc
import os
import pickle
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from metadata import FEAT_COLS
from paths import DUMP_PATH


class DataProcessor:
    """Data processor processing raw data, and providing access to
    processed data ready to be fed into modeling phase.

    Parameters:
       file_path: path of the raw data
           *Note: File reading supports .parquet extension in default
               setting, which can be modified to customized one.
       dp_cfg: hyperparameters of data processor
    """

    def __init__(self, file_path: str, **dp_cfg: Any):
        self._df = pd.read_parquet(file_path)
        self._dp_cfg = dp_cfg
        self._setup()

    def run_before_splitting(self) -> None:
        """Clean and process data before data splitting (i.e., on raw
        static DataFrame).

        Return:
            None
        """
        print("Run data cleaning and processing before data splitting...")

        if self.winsorize_pct:
            self._winsorize()

    def run_after_splitting(
        self,
        df_tr: Union[pd.DataFrame, np.ndarray],
        df_val: Union[pd.DataFrame, np.ndarray],
        fold: int,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], object
    ]:
        """Clean and process data after data splitting to avoid data
        leakage issue.

        Parameters:
            df_tr: training data
            df_val: validation data
            fold: current fold number

        Return:
            df_tr: processed training data
            df_val: processed validation data
            scaler: scaling object
        """
        print("Run data cleaning and processing after data splitting...")
        scaler = None
        if self.trafo_ads["type"] is not None:
            df_tr, df_val, scaler = self._apply_trafo_ads(df_tr, df_val)
            dump_path = os.path.join(DUMP_PATH, f"scalers/fold{fold}.pkl")
            with open(dump_path, "wb") as f:
                pickle.dump(scaler, f)

        return df_tr, df_val, scaler

    def get_df(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return raw or processed DataFrame"""
        return self._df

    def _setup(self) -> None:
        """Retrieve all parameters specified to process data."""
        # Before data splitting
        self.winsorize_pct = self._dp_cfg["winsorize_pct"]
        # After data splitting
        self.trafo_ads = self._dp_cfg["trafo_ads"]

    def _winsorize(self) -> None:
        """Use winsorization to clip feature values exceeding the
        specified limits.
        """
        pct_base = (100 - self.winsorize_pct) / 2

        print(
            f"Winsorize feature values with {self.winsorize_pct}% " "winsorization..."
        )
        for feat in tqdm(FEAT_COLS):
            vals = self._df[feat]
            q1 = vals.quantile(1 - (100 - pct_base) / 100)
            q3 = vals.quantile((100 - pct_base) / 100)
            vals = np.where((vals > q3), q3, vals)
            vals = np.where((vals < q1), q1, vals)
            self._df[feat] = vals
            del vals
            _ = gc.collect()
        print("Done.")

    def _apply_trafo_ads(
        self,
        df_tr: Union[pd.DataFrame, np.ndarray],
        df_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], object
    ]:
        """Transform data after data splitting.

        Parameters:
            df_tr: training data
            df_val: validation data

        Return:
            df_tr: transformed training data
            df_val: transformed validation data
            scaler: scaling object
        """
        if self.trafo_ads["target_only"]:
            cols = ["target"]
        else:
            cols = ["target"] + FEAT_COLS
        n_qts = self.trafo_ads["n_quantiles"]

        if self.trafo_ads["type"] == "qt":
            print(
                f"Transform feature or target with quantile trafo "
                f"using n_quantiles={n_qts}..."
            )
            scaler = QuantileTransformer(
                n_quantiles=n_qts, output_distribution="normal", random_state=42
            )
            df_tr[cols] = scaler.fit_transform(df_tr[cols])
            df_val[cols] = scaler.transform(df_val[cols])

        print("Done")
        return df_tr, df_val, scaler
