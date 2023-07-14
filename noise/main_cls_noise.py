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

# from data.cx14_dataloader import construct_cx14
# from data.openi_dataloader import construct_openi

from data.cx14_dataloader_cut import construct_cx14_cut
from data.cx14_pdc_dataloader_cut import construct_cx14_pdc_cut

from data.openi_dataloader_cut import construct_openi_cut

# from data.openi import construct_loader
from loguru import logger
import wandb
from glob import glob
from utils.utils import *

# from loss.SD_Loss import VAL_LOSS
from loss.val_loss import VAL_LOSS
from models.model import model_zs_sdl
from biobert import BERTEmbedModel
from numpy import linalg as LA
from utils.helper_functions import get_knns, calc_F1
from torch.utils.data import Dataset, DataLoader
import copy
from relabel.sample_graph_maker import relabel_from_sample_graph
from trainer.sgval_trainer import SGVAL

BRED = color.BOLD + color.RED


def config_wandb(args):
    EXP_NAME = "NIH-PDC-NOISY"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    args.run_name = "{}_NOISE_CLS[{}|{}]_MID[mixup={}|beta={}]".format(
        args.run_name,
        args.noise_ratio,
        args.noise_p,
        args.lam,
        args.beta,
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
    # args.batch_size = 64
    # args.num_workers = 8
    # args.use_ensemble = False
    # args.num_classes = 14
    # args.lr_pd = 1e-4
    # args.lr_cls = 0.05

    # args.epochs_cls = 30
    # args.total_runs = 1
    # args.num_pd = 3
    # args.wandb_mode = "online"
    # args.trim_data = True
    # args.run_note = "---"

    logger.bind(stage="CONFIG").critical(
        f"lr_cls = {str(args.lr_cls)} | num_pd = {args.num_pd} | batch_size = {args.batch_size}"
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

    logger.bind(stage="CONFIG").info(f"{args.pd_ckpt}")

    construct_func = construct_cx14_pdc_cut

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

    # ************************************
    # args.load_val_features = True
    # args.load_sample_graph = True
    # # args.pd_ckpt = "[DROP-SAMPLE]RETRAIN SGVAL_20mqqplv"
    # args.pd_ckpt = "run-20220815_204823-xep4835t"
    # args.pd_ckpt = args.pd_ckpt
    # ************************************

    sgval_runner = SGVAL(args, train_loader_val, static_train_loader, train_loader_cls)
    sgval_runner.cls_runner.train_loader.dataset.knn = np.load(
        f"./wandb/{args.pd_ckpt}/files/knn.npy"
    )
    logger.bind(stage="GLOBAL").critical(f"CLS")
    sgval_runner.cls_runner.run()

    wandb.finish()
    return


if __name__ == "__main__":
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
