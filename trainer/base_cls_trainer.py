from abc import abstractmethod
import os, sys
from pyrsistent import v
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
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
from utils.utils import *
from torch.utils.data import Dataset, DataLoader
import copy
import time
from trainer.base_trainer import BASE_TRAINER
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
import scipy.stats as st

st.norm.cdf
st.ttest_1samp


def binarize_vec(v):
    tmp = v.clone()
    tmp[tmp != 0.0] = 1.0
    return tmp


class BASE_CLS(BASE_TRAINER):
    def __init__(self, args, scaler, train_loader) -> None:
        super().__init__(args, scaler, train_loader)

        self.model = densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, args.num_classes)
        self.model = self.model.to(args.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr_cls,
            betas=(0.9, 0.99),
            weight_decay=args.wd_cls,
            eps=0.1,
        )

        # milestone_lr = [0.75 * args.epochs_cls, 0.9 * args.epochs_cls]
        # milestone_lr = [2]
        milestone_lr = [100]
        # self.lr_scheduler = None

        logger.bind(stage="CONFIG").info(f"LR MILESTONE {milestone_lr}")
        self.lr_scheduler = MultiStepLR(self.optim, milestones=milestone_lr, gamma=0.5)
        logger.bind(stage="CONFIG").info(f"LR SCHEDULER {self.lr_scheduler}")

        self.criterion = nn.MultiLabelSoftMarginLoss().to(args.device)
        self.cross_entropy = nn.CrossEntropyLoss().to(args.device)
        self.elr_lambda = args.elr_lambda
        self.smooth_rate = args.lam
        self.b = args.beta

        self.pred_hist = torch.zeros(len(train_loader.dataset), args.num_classes).to(
            args.device
        )
        self.K = 2
        self.pred_beta = 0.9
        self.start_epoch = 0

        self.topk = args.a2s_topk
        logger.bind(stage="CONFIG").info(f"TOPK = {self.topk}")
        if args.resume:
            self._load_from_state_dict()

    def _load_from_state_dict(
        self,
    ):
        assert len(self.args.resume_ckpt) > 0
        ckpt = torch.load(
            os.path.join("./wandb", self.args.resume_ckpt, "files", "model_cls.pth")
        )
        self.start_epoch = ckpt["epoch"]
        self.start_auc = ckpt["mean_auc"]
        logger.bind(stage="LOAD").info(
            f"LOADING CKPT @ epoch {self.start_epoch}, with OpenI mAUC {self.start_auc}"
        )
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])

    def update_hist(self, out, index=None):
        y_pred = torch.sigmoid(out)
        y_pred_ = y_pred.data.detach()
        self.pred_hist[index] = (
            self.pred_beta * self.pred_hist[index] + (1 - self.pred_beta) * y_pred_
        )

    @abstractmethod
    def run(
        self,
    ):
        logger.bind(stage="TRAIN CLS").info("Start Training")
        # lr = self.args.lr_cls

        for epoch in range(self.start_epoch, self.args.epochs_cls):

            mean_auc_openi, mean_auc_pc = 0.0, 0.0   

            self.epoch = epoch

            train_loss = self._train()

            logger.bind(stage="EVAL").warning(
                f"************** Epoch {epoch:04d} Train Loss {train_loss:0.4f} **************"
            )

            if self.args.train_data == "NIH":
                mean_auc_ = self._test_nih()
                auc_pneux, auc_mass_nodule = self._test_nih_google()
            elif self.args.train_data == "CXP":
                mean_auc_ = self._test_cxp()
            else:
                raise ValueError

            if mean_auc_ > self.BEST_AUC_VAL:
                self.BEST_AUC_VAL = mean_auc_
                logger.bind(stage="EVAL").critical("BEST TESTING AUC ON CXP IS UPDATED")

            if self.args.trim_data:
                mean_auc_openi = self._test_openi()
                if self.args.add_noise is False:
                    mean_auc_pc = self._test_pc()
            else:
                logger.bind(stage="EVAL").critical("SKIP OPEN AND PDC EVAL")

            if mean_auc_openi > self.BEST_AUC:
                logger.bind(stage="EVAL").info(
                    f"Update BEST AUC {self.BEST_AUC} >>> {mean_auc_openi}"
                )
                self.BEST_AUC = mean_auc_openi
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "epoch": epoch,
                    "mean_auc_openi": mean_auc_openi,
                    "mean_auc_pc": mean_auc_pc,
                }
                save_checkpoint(state_dict, prefix=f"cls")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            lr_value = self.optim.param_groups[0]["lr"]
            logger.bind(stage="LR DECAY").info(f"LR decay to {lr_value}")
            wandb.log({"LR": lr_value, "epoch": epoch})

        return

    @abstractmethod
    def eval_only(
        self,
    ):
        self.epoch = -1
        # mean_auc_ = self._test_nih_google()
        mean_auc_openi = self._test_openi()
        mean_auc_pc = self._test_pc()
        return

    def loss_gls_org(self, outputs, labels, L=2):
        soft_label = labels * (1 - self.smooth_rate) + self.smooth_rate / L
        a = -(soft_label * labels * F.logsigmoid(outputs))
        b = -(1 - soft_label) * F.logsigmoid(-outputs)
        loss = (a + b).mean()
        # print(f"a {a.mean()}, b {b.mean()}")
        return loss

    def loss_lls(self, p, y_tilde, hist):
        y_bar = (1 - self.smooth_rate) * y_tilde + self.smooth_rate * hist
        loss = self.criterion(p, y_bar)
        return loss

    def loss_gls(self, p, y_tilde, y_knn, hist=0, L=2):
        # y_bar = (
        #     (1 - self.smooth_rate) * y_tilde
        #     + self.smooth_rate * (self.b * (1 + hist) / L + (1 - self.b) * y_knn)
        # ) * binarize_vec(y_knn + y_tilde)

        # no_finding_idx = y_tilde[:,-1]==0
        # loss_ce = self.cross_entropy(p[no_finding_idx], y_tilde[no_finding_idx])
        # y_bar = ((1 - smooth_rate) * y_tilde + smooth_rate * (b * (1 + y_knn) / L + (1 - b) * y_knn))
        y_bar = ((1 - self.smooth_rate) * y_tilde + self.smooth_rate * (y_knn)) * binarize_vec(y_knn + y_tilde)
        loss = self.criterion(p, y_bar)
        return loss

    @abstractmethod
    def _train(
        self,
    ):
        device = self.args.device
        self.model.train()
        total_loss = 0.0

        tbar = tqdm(
            self.train_loader,
            total=int(len(self.train_loader)),
            ncols=100,
            position=0,
            leave=True,
        )

        for batch_idx, (inputs, labels, idx, knn) in enumerate(tbar):
            inputs, labels = inputs.to(device), labels.to(device)
            knn_target = knn.to(device)
            knn_target = knn_target / self.topk

            self.optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(inputs)
                curr_hist = 0
                if self.args.enhance_dist:
                    curr_hist = knn_target.clone()

                if self.args.ols:
                    self.update_hist(outputs.clone(), idx)
                    curr_hist = self.pred_hist[idx].clone()

                loss = self.loss_gls(
                    outputs, labels, knn_target, hist=curr_hist, L=self.K
                )

            total_loss += loss.item()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            # lr_value = self.optim.param_groups[0]["lr"]

            wandb.log({"CLS Loss": loss, "epoch": self.epoch})
        return total_loss / (batch_idx + 1)

    @abstractmethod
    def _test(self, test_loader, net2=None, clean_nih=False):
        self.model.eval()
        if net2 is not None:
            net2.eval()

        all_preds, all_gts = [], []
        total_loss = 0.0
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
                preds = torch.sigmoid(outputs)
                all_preds.append(preds)
                all_gts.append(labels)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_gts = torch.cat(all_gts).cpu().numpy()

        if clean_nih:
            pred_pneux = all_preds[:, [7]]
            pred_mass_nodule = np.concatenate(
                (all_preds[:, [4]], all_preds[:, [5]]), axis=1
            )
            assert (all_gts[:, 4] - all_gts[:, 5]).sum() == 0.0
            auc_pneux = roc_auc_score(all_gts[:, 7], pred_pneux[:, 0])
            auc_mass_nodule = roc_auc_score(all_gts[:, 4], pred_mass_nodule.mean(-1))
            return auc_pneux, auc_mass_nodule

        all_auc = [
            roc_auc_score(all_gts[:, i], all_preds[:, i])
            for i in range(self.num_classes - 1)
        ]
        all_map = [
            average_precision_score(all_gts[:, i], all_preds[:, i])
            for i in range(self.num_classes - 1)
        ]
        return all_auc, total_loss / (batch_idx + 1), all_map
