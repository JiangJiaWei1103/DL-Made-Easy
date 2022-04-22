"""
Common utility functions used in training and evaluation processes.
Author: JiaWei Jiang
"""
import os
from decimal import Decimal
from typing import Any, Dict

import yaml
from torch.nn import Module

import wandb
from paths import DUMP_PATH


def count_params(model: Module) -> str:
    """Count number of parameters in model.

    Parameters:
        model: model instance

    Return:
        n_params: number of parameters in model, represented in
            scientific notation
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = f"{Decimal(str(n_params)):.4E}"

    return n_params


def dump_wnb(project_name: str, cfg: Dict[str, Any], exp_id: str) -> None:
    """Dump and push experiment output objects.

    Parameters:
        project_name: name of the project
        cfg: configuration of entire experiment
        exp_id: experiment identifier

    Return:
        None
    """
    cfg_dump_path = os.path.join(DUMP_PATH, "cfg.yaml")
    with open(cfg_dump_path, "w") as f:
        yaml.dump(cfg, f)
    dump_entry = wandb.init(project=project_name, group=exp_id, job_type="dumping")
    model_name = cfg["common"]["model_name"]
    artif = wandb.Artifact(name=model_name.upper(), type="output")
    artif.add_dir(DUMP_PATH)
    dump_entry.log_artifact(artif)  # type: ignore
    # (tmp. workaround)
    wandb.finish()
