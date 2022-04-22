"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Moreover, evaluation on unseen (testing) data is
optional.
"""
import gc
import warnings
from argparse import Namespace

import pandas as pd

import wandb
from config.config import gen_exp_id, setup_dp, setup_model, setup_proc
from criterion.build import build_criterion
from cv.build import build_cv_iter
from data.build import build_dataloaders
from data.data_processor import DataProcessor
from engine.defaults import DefaultArgParser
from evaluating.build import build_evaluator
from modeling.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import MainTrainer
from utils.common import count_params, dump_wnb
from utils.early_stopping import EarlyStopping

warnings.simplefilter("ignore")


def main(args: Namespace) -> None:
    """Run training and evaluation processes.

    Parameters:
        args: arguments driving training and evaluation processes

    Returns:
        None
    """
    # Configure experiment
    project_name = args.project_name
    input_path = args.input_path
    model_name = args.model_name
    exp_id = gen_exp_id(model_name)
    dp_cfg = setup_dp()
    proc_cfg = setup_proc()
    model_cfg = setup_model(model_name)
    cfg = {
        "common": vars(args),
        "dp": dp_cfg,
        "proc": proc_cfg,
        "model": model_cfg,
    }
    device = proc_cfg["device"]
    eval_metrics = proc_cfg["evaluator"]["eval_metrics"]

    # Clean and process data to feed
    dp = DataProcessor(input_path, **dp_cfg)
    dp.run_before_splitting()
    df = dp.get_df()

    # Start cv process
    cv_iter = build_cv_iter(df, args)
    for i, (tr_idx, val_idx) in enumerate(cv_iter):
        # Configure experiment
        exp = wandb.init(
            project=project_name,
            config=cfg,
            group=exp_id,
            job_type="train_eval",
            name=f"fold{i}",
        )
        print(f"Training and evaluation process of fold{i} starts...")

        # Build dataloaders
        # Further data processing to take care of leakage issue
        if isinstance(df, pd.DataFrame):
            df_tr, df_val = df.iloc[tr_idx, :], df.iloc[val_idx, :]
        else:
            df_tr, df_val = df[tr_idx, :], df[val_idx, :]
        df_tr, df_val, scaler = dp.run_after_splitting(df_tr, df_val, i)
        train_loader, val_loader = build_dataloaders(
            df_tr,
            df_val,
            model_name,
            **proc_cfg["dataloader"],
            **dp_cfg["dataset"],
        )

        # Build model
        model = build_model(model_name, model_cfg)
        wandb.log({"model": {"n_params": count_params(model)}})
        model.to(device)
        wandb.watch(model, log="all", log_graph=True)

        # Build criterion
        loss_fn = build_criterion(**proc_cfg["loss_fn"])

        # Build solvers
        optimizer = build_optimizer(model, **proc_cfg)
        lr_skd = build_lr_scheduler(optimizer, **proc_cfg)

        # Build early stopping tracker
        if proc_cfg["patience"] != 0:
            es = EarlyStopping(proc_cfg["patience"], proc_cfg["mode"])
        else:
            es = None

        # Build evaluator
        evaluator = build_evaluator(eval_metrics)

        # Run main training and evaluation for one fold
        trainer = MainTrainer(
            proc_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            es,
            evaluator,
            train_loader,
            eval_loader=val_loader,
            scaler=scaler,
        )
        trainer.train_eval(i)

        # Free mem.
        del (
            df_tr,
            df_val,
            train_loader,
            val_loader,
            model,
            optimizer,
            lr_skd,
            es,
            evaluator,
            trainer,
        )
        _ = gc.collect()

    # Dump cfg and outputs
    dump_wnb(project_name, cfg, exp_id)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = DefaultArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
