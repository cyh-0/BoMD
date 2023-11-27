import os
import sys
from glob import glob

import numpy as np
import torch
import wandb
from loguru import logger

from data.cx14_dataloader_cut import construct_cx14_cut
from data.cx14_pdc_dataloader_cut import construct_cx14_pdc_cut
from data.cxp_dataloader_cut import construct_cxp_cut
from opts import parse_args
from trainer.bomd_trainer import BoMD
from utils.utils import *

BRED = color.BOLD + color.RED


def config_wandb(args):
    EXP_NAME = "EVAL-CKPT"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    args.run_name = (
        "{}_BASE[{}|{}]_A2S[func={}|topk={}|drop-th={}|mixup={}|beta={}]".format(
            args.run_name,
            args.batch_size,
            args.lr_cls,
            args.relabel_method,
            args.nsd_topk,
            args.nsd_drop_th,
            args.lam,
            args.beta,
        )
    )
    wandb.init(project=EXP_NAME, notes=args.run_note, name=args.run_name)

    log_file = open(os.path.join(wandb.run.dir, "loguru.txt"), "w")
    logger.add(log_file, enqueue=True)
    # wandb.run.name = wandb.run.id
    config = wandb.config
    config.update(args)

    logger.bind(stage="CONFIG").critical("WANDB_MODE = {}".format(args.wandb_mode))
    logger.bind(stage="CONFIG").info("Experiment Name: {}".format(EXP_NAME))
    return


def load_args():
    args = parse_args()
    logger.bind(stage="CONFIG").critical(
        f"lr_cls = {str(args.lr_cls)} | num_fea = {args.num_fea} | batch_size = {args.batch_size}"
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

    if args.train_data == "NIH":
        construct_func = construct_cx14_cut
    else:
        construct_func = construct_cxp_cut

    """
    NIH
    """
    train_loader_val = construct_func(
        args, mode="train", file_name="train", stage="MID"
    )
    static_train_loader = construct_func(
        args, mode="test", file_name="train", stage="STATIC"
    )
    train_loader_cls = construct_func(
        args, mode="train", file_name="train", stage="CLS"
    )

    args.mid_ckpt = "run-20220816_175557-6dn2k0id"

    sgval_runner = BoMD(args, train_loader_val, static_train_loader, train_loader_cls)

    knn = np.load(f"./ckpt/saved/{args.mid_ckpt}/files/knn.npy")

    sgval_runner.cls_runner.train_loader.dataset.knn = knn

    logger.bind(stage="GLOBAL").critical(f"CLS")

    ckpt_fp = "./ckpt/BEST_NIH/model_cls.pth"
    ckpt = torch.load(ckpt_fp)
    sgval_runner.cls_runner.model.load_state_dict(ckpt["model"])
    sgval_runner.cls_runner.epoch = ckpt["epoch"]
    sgval_runner.cls_runner.eval_only()

    wandb.finish()
    return


if __name__ == "__main__":
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
