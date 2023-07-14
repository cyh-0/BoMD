import os, sys
from torch._C import device
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.nn as nn
from opts import parse_args
import torchvision
from models.densenet import densenet121

# from data.cx14_dataloader_cut import construct_cx14_cut
# from data.openi_dataloader_cut import construct_openi_cut
# from data.padchest_dataloader_cut import construct_pc_cut

from data.cx14_dataloader_cut import construct_cx14_cut
from data.openi_dataloader_cut import construct_openi_cut
from data.padchest_dataloader_cut import construct_pc_cut
from data.cxp_dataloader_cut import construct_cxp_cut

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

BRED = color.BOLD + color.RED


def config_wandb(args):
    EXP_NAME = "CXP-CX-(8)"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project=EXP_NAME)
    wandb.run.name = wandb.run.id
    config = wandb.config
    config.update(args)
    # if args.wandb_mode == "online":
    #     code = wandb.Artifact("project-source", type="code")
    #     for path in glob("*.py", recursive=True):
    #         code.add_file(path)
    #     wandb.run.use_artifact(code)
    logger.bind(stage="CONFIG").critical("WANDB_MODE = {}".format(args.wandb_mode))
    logger.bind(stage="CONFIG").info("Experiment Name: {}".format(EXP_NAME))
    return


def load_args():
    args = parse_args()
    # args.batch_size = 32
    # args.num_workers = 12
    # args.use_ensemble = False
    # args.trim_data = True
    # args.num_classes = 8
    # args.lr_pd = 1e-4
    # # args.lr_cls = 0.05
    # args.epochs_cls = 30
    # args.total_runs = 1
    # args.wandb_mode = "online"
    # args.num_pd = 3
    # args.train_data = "CXP"
    # args.run_note = ""
    logger.bind(stage="CONFIG").critical(
        f"train_data = {str(args.train_data)} | num_pd = {args.num_pd} | image_size = {args.resize}"
    )
    return args


def load_word_vec(args):
    wordvec_array = np.load("./embeddings/cxp_biobert_8.npy")

    normed_wordvec = np.stack(
        [
            wordvec_array[i] / LA.norm(wordvec_array[i])
            for i in range(wordvec_array.shape[0])
        ]
    )
    normed_wordvec = (
        (torch.tensor(normed_wordvec).to(args.device).float())
        .permute(1, 0)
        .unsqueeze(0)
    )
    return normed_wordvec


def get_word_vec(model, pathology):
    tmp_list = []
    for i in pathology:
        current = model.get_embeds(i)
        tmp_list.append(current)
    saved_embd = torch.stack(tmp_list).numpy()
    return saved_embd


from relabel.ralabel_main import RELABEL_MAIN


def main():
    BEST_AUC = -np.inf
    args = load_args()
    config_wandb(args)
    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        logger.bind(stage="CONFIG").debug(error)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    wordvec_array = load_word_vec(args)

    train_loader = construct_cxp_cut(
        args, args.root_dir, mode="train", file_name="train"
    )

    static_train_loader = construct_cxp_cut(
        args, args.root_dir, mode="test", file_name="train"
    )
    for iter in range(args.total_runs):
        logger.bind(stage="GLOBAL").critical(f"Global Iteration {iter}")
        # train_loader = TRAIN_PD(args, wordvec_array, scaler, train_loader)
        new_gt = RELABEL_MAIN(args, static_train_loader, wordvec_array, iter)
        train_loader.dataset.gt = new_gt.copy()
        static_train_loader.dataset.set = new_gt.copy()
        TRAIN_CLS(args, wordvec_array, scaler, train_loader)
        print("\n ===================================================== \n")
    return


if __name__ == "__main__":

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
