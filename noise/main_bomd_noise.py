from relabel.ralabel_main import RELABEL_MAIN
import os
import sys
from torch._C import device
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.nn as nn
from opts import parse_args
import torchvision
from models.densenet import densenet121
from data.cx14_dataloader_cut import construct_cx14_cut
from data.cx14_pdc_dataloader_cut import construct_cx14_pdc_cut


# from data.openi import construct_loader
from loguru import logger
import wandb
from utils.utils import *
from trainer.sgval_trainer import SGVAL

BRED = color.BOLD + color.RED


def config_wandb(args):
    EXP_NAME = "NIH-PDC-NOISY"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    args.run_name = "{}_NOISE[{}|{}]_MID[mixup={}|beta={}]".format(
        args.run_name,
        args.noise_ratio,
        args.noise_p,
        args.lam,
        args.beta,
    )
    wandb.init(project=EXP_NAME, notes=args.run_note, name=args.run_name)
    config = wandb.config
    config.update(args)
    logger.bind(stage="CONFIG").critical("NAME = {}".format(args.run_name))
    logger.bind(stage="CONFIG").critical("WANDB_MODE = {}".format(args.wandb_mode))
    logger.bind(stage="CONFIG").info("Experiment Name: {}".format(EXP_NAME))
    return


def load_args():
    args = parse_args()
    logger.bind(stage="CONFIG").critical(
        f"use_ensemble = {str(args.use_ensemble)} || num_pd = {args.num_pd}"
    )
    return args


def main():
    args = load_args()
    print(args)
    config_wandb(args)
    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        logger.bind(stage="CONFIG").debug(error)

    log_file = open(os.path.join(wandb.run.dir, "loguru.txt"), "w")
    logger.add(log_file, enqueue=True)

    if args.add_noise:
        construct_func = construct_cx14_pdc_cut
    else:
        construct_func = construct_cx14_cut

    """
    NIH
    """
    train_loader_val = construct_func(
        args, mode="train", file_name="train", stage="VAL"
    )
    static_train_loader = construct_func(
        args, mode="test", file_name="train", stage="STATIC"
    )
    train_loader_cls = construct_func(
        args, mode="train", file_name="train", stage="CLS"
    )

    static_train_loader.dataset.train_label = train_loader_val.dataset.train_label
    train_loader_cls.dataset.train_label = train_loader_val.dataset.train_label

    sanity_check(train_loader_val, static_train_loader, train_loader_cls)

    args.load_val_features = True
    args.load_sample_graph = True

    sgval_runner = SGVAL(args, train_loader_val, static_train_loader, train_loader_cls)
    # sgval_runner.deploy()

    # Start from relabel 
    sgval_runner.a2s_runner.run()
    sgval_runner.cls_runner.train_loader.dataset.knn = np.load(
        os.path.join(wandb.run.dir, "knn.npy")
    )
    print(sgval_runner.cls_runner.train_loader.dataset.knn)
    # sgval_runner.cls_runner.train_loader.dataset.knn = np.load(
    #     os.path.join("./wandb", args.pd_ckpt, "files", "knn.npy")
    # )
    logger.bind(stage="GLOBAL").critical(f"CLS")
    sgval_runner.cls_runner.run()

    wandb.finish()
    return


if __name__ == "__main__":

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
