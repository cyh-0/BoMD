from abc import abstractmethod
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

from torch.optim import lr_scheduler

# from data.openi import construct_loader
from loguru import logger
import wandb
from glob import glob
from utils.utils import *
from torch.utils.data import Dataset, DataLoader
import copy
import time
from trainer.base_trainer import BASE_TRAINER
from models.model import model_mid
from loss.val_loss import VAL_LOSS
from utils.helper_functions import get_knns, calc_F1
from data.cxp_dataloader_cut import construct_cxp_cut
from data.cx14_dataloader_cut import construct_cx14_cut


class VAL:
    def __init__(self, args, scaler, wordvec_array, train_loader) -> None:
        self.BEST_AUC = -np.inf

        self.args = args
        self.scaler = scaler
        self.device = args.device
        self.train_loader = train_loader
        self.wordvec_array = wordvec_array

        self.model = model_mid(args).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=args.lr_pd, weight_decay=0
        )
        self.scheduler = lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=args.lr_pd,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs_val,
            pct_start=0.1,
        )
        self.mid_criteria = VAL_LOSS(
            wordvec_array=wordvec_array, weight=0.3, args=args
        ).to(self.device)

        self.test_loader_nih = construct_cx14_cut(
            args, mode="test", file_name="test", stage="VAL"
        )
        self.test_loader_cxp = construct_cxp_cut(args, mode="test", file_name="test")

    @abstractmethod
    def run(
        self,
    ):
        self.model.train()
        # word_vecs = wordvec_array.squeeze().transpose(0, 1)
        for epoch in range(self.args.epochs_val):
            total_loss = 0.0
            tbar = tqdm(
                self.train_loader,
                total=int(len(self.train_loader)),
                ncols=100,
                position=0,
                leave=True,
            )
            for batch_idx, (inputs, labels, item) in enumerate(tbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(inputs)
                    loss = self.mid_criteria(outputs, labels)
                total_loss += loss.item()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()
                lr_value = self.optim.param_groups[0]["lr"]
                wandb.log({"Learning Rate": lr_value, "SD_Loss": loss})

            logger.bind(stage="TRAIN").info(
                f"Epoch {epoch} Train Loss:{total_loss / (batch_idx + 1):0.4f}"
            )

            if self.args.train_data == "NIH":
                mean_auc = self._test_nih()
            elif self.args.train_data == "CXP":
                mean_auc = self._test_cxp()
            else:
                raise ValueError

            if mean_auc > self.BEST_AUC:
                self.BEST_AUC = mean_auc
                state_dict = {
                    "net": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "epoch": epoch,
                    "mean_auc": mean_auc,
                    # "all_auc": np.asarray(all_auc),
                }
                save_checkpoint(state_dict, prefix="best_pd")

        return

    @abstractmethod
    def _test(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        word_vecs = self.wordvec_array.squeeze().transpose(0, 1)
        preds_regular = []
        targets = []

        tbar = tqdm(
            test_loader,
            total=int(len(test_loader)),
            ncols=100,
            position=0,
            leave=True,
        )

        for batch_idx, (inputs, labels, item) in enumerate(tbar):
            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.mid_criteria(outputs, labels)
                total_loss += loss.item()

                preds_regular.append(outputs.cpu().detach())
                targets.append(labels.cpu().detach())

        idxs, dists = get_knns(
            word_vecs.cpu().detach(), torch.cat(preds_regular).numpy()
        )

        precision_3, recall_3, F1_3 = calc_F1(
            torch.cat(targets).numpy(),
            idxs,
            self.args.num_pd,
            num_classes=len(word_vecs),
        )
        test_loss = total_loss / (batch_idx + 1)

        return precision_3, recall_3, F1_3, test_loss

    @abstractmethod
    def _test_nih(
        self,
    ):
        logger.bind(stage="EVAL").info("************** EVAL ON NIH **************")
        precision, recall, F1, test_loss = self._test(self.test_loader_nih)
        wandb.log(
            {
                "NIH Test Loss": test_loss,
                "NIH F1_score": F1,
                "NIH Precision": precision,
                "NIH Recall": recall,
            }
        )
        logger.bind(stage="EVAL").success(
            f"Test Loss:{test_loss:0.4f}    F1_score:{F1:0.4f}    Precision:{precision:0.4f}    Recall:{recall:0.4f}"
        )

        return F1

    @abstractmethod
    def _test_cxp(
        self,
    ):
        logger.bind(stage="EVAL").info("************** EVAL ON CXP **************")
        precision, recall, F1, test_loss = self._test(self.test_loader_cxp)
        wandb.log(
            {
                "CXP Test Loss": test_loss,
                "CXP F1_score": F1,
                "CXP Precision": precision,
                "CXP Recall": recall,
            }
        )
        logger.bind(stage="EVAL").success(
            f"Test Loss:{test_loss:0.4f}    F1_score:{F1:0.4f}    Precision:{precision:0.4f}    Recall:{recall:0.4f}"
        )
        return F1
