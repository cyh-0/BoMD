from abc import abstractmethod
from ast import Pass
import os
import sys
from pyrsistent import v
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

from data.cxp_dataloader_cut import construct_cxp_cut
from data.cx14_dataloader_cut import construct_cx14_cut
from data.openi_dataloader_cut import construct_openi_cut
from data.padchest_dataloader_cut import construct_pc_cut

# from data.openi import construct_loader
from loguru import logger
import wandb
from glob import glob
from utils.utils import log_csv
from torch.utils.data import Dataset, DataLoader
import copy
import time


class BASE_TRAINER:
    def __init__(self, args, scaler, train_loader):
        self.args = args
        self.BEST_AUC = -np.inf
        self.BEST_AUC_VAL = -np.inf

        self.scaler = scaler
        self.device = args.device

        self.num_classes = args.num_classes

        self.train_loader = train_loader
        # self.TRAIN_SIZE = train_loader.__len__()

        self.test_loader_nih = construct_cx14_cut(args, mode="test", file_name="test")
        self.test_loader_nih_google = construct_cx14_cut(
            args, mode="test", file_name="clean_test"
        )
        self.test_loader_openi = construct_openi_cut(args, mode="test")
        self.test_loader_padchest = construct_pc_cut(args, mode="test")
        self.test_loader_cxp = construct_cxp_cut(args, mode="test", file_name="test")

        # self.optim = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=args.lr_cls,
        #     betas=(0.9, 0.99),
        #     weight_decay=args.wd_cls,
        #     eps=0.1,
        # )

    # @abstractmethod
    # def run(self,):
    #     pass

    @abstractmethod
    def _train(
        self,
    ):
        pass

    @abstractmethod
    def _test(self, test_loader, net2=None):
        pass

    def _test_nih(
        self,
    ):
        logger.bind(stage="EVAL").info("************** EVAL ON NIH **************")
        all_auc, test_loss, all_map = self._test(self.test_loader_nih)
        mean_auc = np.asarray(all_auc).mean()
        wandb.log({"Eval/MeanAUC NIH": mean_auc, "epoch": self.epoch})
        logger.bind(stage="EVAL").success(f"Mean AUC NIH {mean_auc:0.4f}")
        return mean_auc

    def _test_nih_google(
        self,
    ):
        logger.bind(stage="EVAL").info(
            "************** EVAL ON NIH GOOGLE **************"
        )
        auc_pneux, auc_mass_nodule = self._test(
            self.test_loader_nih_google, clean_nih=True
        )

        wandb.log(
            {
                "NIH GOOGLE AUC/Mass or Nodule": auc_mass_nodule,
                "NIH GOOGLE AUC/Pheumothorax": auc_pneux,
                "epoch": self.epoch,
            }
        )
        logger.bind(stage="EVAL").success(
            f"NIH GOOGLE AUC "
            f"Mass/Nodule: {auc_mass_nodule:0.4f} "
            f"Pheumothorax: {auc_pneux:0.4f} ",
        )
        return auc_pneux, auc_mass_nodule

    def _test_openi(
        self,
    ):
        logger.bind(stage="EVAL").info("************** EVAL ON OPENI **************")
        all_auc, test_loss, all_map = self._test(self.test_loader_openi)
        mean_auc = np.asarray(all_auc).mean()
        mean_map = np.asarray(all_map).mean()
        wandb.log({"Eval/MeanAUC OPENI": mean_auc, "epoch": self.epoch})
        logger.bind(stage="EVAL").success(
            f"Mean AUC OPENI {mean_auc:0.4f} mAP OPENI {mean_map:0.4f}"
        )
        log_csv(self.epoch, all_auc, mean_auc, mean_map, "openi_auc")
        return mean_auc

    def _test_pc(
        self,
    ):
        logger.bind(stage="EVAL").info("************** EVAL ON PADCHEST **************")
        all_auc, test_loss, all_map = self._test(self.test_loader_padchest)
        mean_auc = np.asarray(all_auc).mean()
        mean_map = np.asarray(all_map).mean()
        wandb.log({"Eval/MeanAUC PADCHEST": mean_auc, "epoch": self.epoch})
        logger.bind(stage="EVAL").success(
            f"Mean AUC PADCHEST {mean_auc:0.4f} mAP PADCHEST {mean_map:0.4f}"
        )
        log_csv(self.epoch, all_auc, mean_auc, mean_map, "padchest_auc")
        return mean_auc

    def _test_cxp(
        self,
    ):
        logger.bind(stage="EVAL").info("************** EVAL ON CXP **************")
        all_auc, test_loss, all_map = self._test(self.test_loader_cxp)
        mean_auc = np.asarray(all_auc).mean()
        mean_map = np.asarray(all_map).mean()
        wandb.log({"Eval/MeanAUC CXP": mean_auc, "epoch": self.epoch})
        logger.bind(stage="EVAL").success(
            f"Mean AUC CXP {mean_auc:0.4f} mAP CXP {mean_map:0.4f}"
        )
        return mean_auc
