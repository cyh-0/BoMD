from abc import abstractmethod
import os
import sys
from pyrsistent import v
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from loguru import logger
import wandb
from glob import glob
from utils.utils import *
from trainer.base_cls_trainer import BASE_CLS
from trainer.val_trainer import VAL
from numpy import linalg as LA
from utils.helper_functions import get_knns, calc_F1
from models.model import model_zs_sdl
from loss.val_loss import VAL_LOSS
import faiss
from utils.utils import load_word_vec


class SGVAL:
    def __init__(
        self, args, train_loader, static_train_loader, train_loader_cls
    ) -> None:

        self.args = args
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.wordvec_array = load_word_vec(args)

        self.val_runner = VAL(args, self.scaler, self.wordvec_array, train_loader)
        self.a2s_runner = A2S(args, self.wordvec_array, static_train_loader)
        self.cls_runner = BASE_CLS(args, self.scaler, train_loader_cls)

    @abstractmethod
    def deploy(
        self,
    ):
        """
        Pre-training
        """
        logger.bind(stage="GLOBAL").critical(f"VAL")
        self.val_runner.run()
        torch.cuda.empty_cache()
        """
        KNN Graph
        """
        logger.bind(stage="GLOBAL").critical(f"A2S")
        self.a2s_runner.run()
        """
        Classification
        """
        self.cls_runner.train_loader.dataset.knn = np.load(
            os.path.join(wandb.run.dir, "knn.npy")
        )
        logger.bind(stage="GLOBAL").critical(f"CLS")
        self.cls_runner.run()
        return


class A2S:
    def __init__(self, args, wordvec_array, static_train_loader) -> None:
        self.args = args
        self.embed_len = args.embed_len
        self.num_classes = args.num_classes
        self.device = self.args.device
        # self.train_loader = train_loader
        self.static_train_loader = static_train_loader
        self.wordvec_array = wordvec_array
        self.model = model_zs_sdl(self.args)
        self.org_gt = self.static_train_loader.dataset.train_label
        self.train_size = self.org_gt.shape[0]
        self.relabel_methods = f"relabel_v{args.relabel_method}"
        self.relabel_func = getattr(self, self.relabel_methods)
        logger.bind(stage="A2S").warning(f"Using method v{args.relabel_method}")

        if self.args.load_val_features:
            # fp = "run-20220124_140318-24mckzen"
            fp = self.args.pd_ckpt
            logger.bind(stage="A2S").warning("LOADING PD CKPT")
            # pred_idxs = np.load("./relabel/run-20220124_140318-24mckzen_pred_rank.npy")
            # va_matrix = np.load("./relabel/run-20220124_140318-24mckzen_matrix.npy")
            self.pred_idxs = np.load(f"./wandb/{fp}/files/pred_rank.npy")
            self.va_matrix = np.load(f"./wandb/{fp}/files/va_matrix.npy")
            # self.pred_idxs = self.pred_idxs[-14714:]
            # self.va_matrix = self.va_matrix[-14714:]

        if self.args.load_sample_graph:
            logger.bind(stage="A2S").warning("LOADING SAMPLE GRAPH")
            fp = self.args.pd_ckpt
            self.sample_graph = np.load(f"./wandb/{fp}/files/sample_graph.npy")

    def run(
        self,
    ):
        if not self.args.load_val_features:
            logger.bind(stage="A2S").critical(f"FETCHING ATTRIBUTES")
            self.load_model()
            self.pred_idxs, self.va_matrix = self.get_attributes()

        if not self.args.load_sample_graph:
            logger.bind(stage="A2S").critical(f"CONSTRUCTING SAMPLE GRAPH")
            self.sample_graph = self.graph_based_maker()

        self.knn_relabel()
        return

    def load_model(
        self,
    ):
        if len(self.args.pd_ckpt) != 0:
            tmp = self.args.pd_ckpt
            dir = f"./wandb/{tmp}/files"
        else:
            dir = wandb.run.dir
        fp = os.path.join(dir, "model_best_pd.pth")
        self.model.load_state_dict(torch.load(fp)["net"])
        self.model.to(self.device)

    def get_attributes(
        self,
    ):
        # train_loader = construct_cx14(args, args.root_dir, mode="god", file_name="train")
        sd_criteria = VAL_LOSS(
            wordvec_array=self.wordvec_array, weight=0.3, args=self.args
        ).to(self.device)
        total_loss = 0.0
        preds_regular = []
        targets = []
        self.wordvec_array = self.wordvec_array.squeeze().transpose(0, 1)
        self.model.eval()

        tbar = tqdm(
            self.static_train_loader,
            total=int(len(self.static_train_loader)),
            ncols=100,
            position=0,
            leave=True,
        )

        with torch.no_grad():
            for batch_idx, (inputs, labels, idx) in enumerate(tbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = sd_criteria(outputs, labels)
                total_loss += loss.item()

                preds_regular.append(outputs.unsqueeze(1).cpu().detach())
                targets.append(labels.cpu().detach())
                wandb.log({"relabel_loss": loss})

        # np.save("./pred_pd.npy", torch.cat(preds_regular).numpy())

        train_loss = total_loss / (batch_idx + 1)
        logger.bind(stage="A2S").info(f"Train Loss {train_loss}")

        va_matrix = torch.cat(preds_regular).numpy()
        pred_idxs, dists = get_knns(self.wordvec_array.cpu().detach(), va_matrix)

        np.save(
            os.path.join(wandb.run.dir, "pred_rank.npy"),
            pred_idxs,
        )

        np.save(
            os.path.join(wandb.run.dir, "va_matrix.npy"),
            va_matrix,
        )
        return pred_idxs, va_matrix

    def graph_based_maker(
        self,
    ):
        num_pd = self.args.num_pd

        # pd_reshape = va_matrix
        print(f"number of pd :{num_pd}")
        pd_reshape = self.va_matrix.reshape(
            self.va_matrix.shape[0], self.embed_len, num_pd
        ).transpose(0, 2, 1)
        pd_reshape = pd_reshape.reshape(-1, self.embed_len)
        # pd_reshape

        topk = 200
        index = faiss.IndexFlatL2(
            pd_reshape.shape[-1]
        )  # OR IP for cosine similarity. But need to normalize before clustering
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(pd_reshape)

        D, I = gpu_index.search(pd_reshape, topk)
        # Attribute-level Graph
        idx = I.reshape(self.train_size, num_pd, topk)

        # args = edict({"device": 0})
        # self.wordvec_array = load_word_vec(args)
        # self.wordvec_array = self.wordvec_array.squeeze().transpose(0, 1)

        sample_graph = []
        # Sample-level Graph
        for sample_idx in tqdm(range(self.train_size), ncols=100):
            lb_corr = [
                np.unique(idx[sample_idx, i, :] // num_pd, return_index=True)
                for i in range(num_pd)
            ]
            sort_index0 = None
            sort_index0 = [
                lb_corr[i][0][np.argsort(lb_corr[i][1])] for i in range(num_pd)
            ]
            sort_index0 = [i.reshape(1, -1) for i in sort_index0]
            idx_score = []
            uniq_sample = np.unique(np.concatenate(sort_index0, axis=1))
            for item in uniq_sample:
                tmp_list = []
                for row in sort_index0:
                    tmp_list.extend(np.where(row == item)[1])
                idx_score.append(tmp_list)

            argmax_idx = np.argsort([sum(i) for i in idx_score])
            # Top 20 sample
            sorted_sample = uniq_sample[argmax_idx][1:51]
            sample_graph.append(sorted_sample)

        sample_graph = np.stack(sample_graph, axis=1).transpose(1, 0)

        np.save(os.path.join(wandb.run.dir, "sample_graph.npy"), sample_graph)
        return sample_graph
        # sample_graph = np.load("./clustering_test/sample_graph.npy")

    def knn_relabel(
        self,
    ):
        self.sample_graph = self.sample_graph[:, : self.args.a2s_topk]
        logger.bind(stage="A2S").critical(f"TOP-{self.args.a2s_topk} LABEL AGGREGATION")

        total_new_label = []
        # pred = np.zeros((self.num_classes,))
        self.CLEAN_LIST = []
        for top_num in range(1, 7):
            for i in tqdm(range(self.train_size), ncols=100):
                curret_pred = np.zeros((self.num_classes,))
                pos_idx = self.pred_idxs[i][:top_num]
                curret_pred[pos_idx] = 1

                if (self.org_gt[i] == curret_pred).all():
                    # pred += curret_pred
                    self.CLEAN_LIST.append(i)
        self.CLEAN_LIST = sorted(self.CLEAN_LIST)
        logger.bind(stage="A2S").info(
            f"CLEAN SET CONTAIN {len(self.CLEAN_LIST)} SAMPLES"
        )

        for sample_idx in tqdm(range(self.train_size), ncols=100):
            row_new_label = self.relabel_func(sample_idx)
            total_new_label.append(row_new_label)

        new_gt = np.stack(total_new_label)
        np.save(os.path.join(wandb.run.dir, "knn.npy"), new_gt)
        # np.save(os.path.join(wandb.run.dir, "relabelled_gt.npy"), new_gt)
        # np.save(os.path.join("./wandb/run-20220804_022002-1e2g3ooa/files", "relabelled_gt_v2.npy"), new_gt)
        return new_gt

    # Purely label aggregation
    def relabel_v1(self, sample_idx):
        row_new_label = None
        if sample_idx in self.CLEAN_LIST:
            row_new_label = self.org_gt[sample_idx].copy()

        else:
            topk_mat = np.stack(
                [self.org_gt[sp_idx] for sp_idx in self.sample_graph[sample_idx]]
            )
            # Top 1 is no finding, assign no finding
            if topk_mat[0][-1] == 1.0 and np.sum(topk_mat, axis=0)[-1] >= 8:
                row_new_label = np.zeros((self.num_classes,))
                row_new_label[-1] = 1
                # print("activation no finding")
            else:
                sum_topk = np.sum(topk_mat, axis=0)
                sum_topk[-1] = 0
                sum_topk[sum_topk > 0.0] = 1
                # if sum_topk.sum() == 0.0:
                #     sum_topk[-1] = 1
                row_new_label = self.org_gt[sample_idx] * self.args.lam + sum_topk * (
                    1 - self.args.lam
                )
        return row_new_label

    # Try to filter out low evidenced samples
    def relabel_v2(self, sample_idx):
        row_new_label = None
        if sample_idx in self.CLEAN_LIST:
            row_new_label = self.org_gt[sample_idx].copy()
        else:
            topk_mat = np.stack(
                [self.org_gt[sp_idx] for sp_idx in self.sample_graph[sample_idx]]
            )
            # Top 1 is no finding, assign no finding
            # if topk_mat[0][-1] == 1.0 and np.sum(topk_mat, axis=0)[-1] >= 4:
            #     row_new_label = np.zeros((self.num_classes,))
            #     row_new_label[-1] = 0
            #     # print("activation no finding")
            # else:
            sum_topk = np.sum(topk_mat, axis=0)
            if np.argmax(sum_topk) == 13:
                row_new_label = np.zeros((self.num_classes,))
                row_new_label[-1] = 1
                return row_new_label
            else:
                sum_topk[-1] = 0
                sum_topk[sum_topk < self.args.a2s_drop_th] = 0
                sum_topk[sum_topk != 0.0] = 1
                if sum_topk.sum() == 0.0:
                    sum_topk[-1] = 1
                    return sum_topk
                row_new_label = self.org_gt[sample_idx] * self.args.lam + sum_topk * (
                    1 - self.args.lam
                )
        return row_new_label

    def relabel_v3(self, sample_idx):
        row_new_label = None
        if sample_idx in self.CLEAN_LIST:
            row_new_label = self.org_gt[sample_idx].copy() * self.args.a2s_topk
        else:
            topk_mat = np.stack(
                [self.org_gt[sp_idx] for sp_idx in self.sample_graph[sample_idx]]
            )
            row_new_label = np.sum(topk_mat, axis=0)
            # return
            # # Top 1 is no finding, assign no finding
            # if topk_mat[0][-1] == 1.0 and np.sum(topk_mat, axis=0)[-1] >= 8:
            #     row_new_label = np.zeros((self.num_classes,))
            #     row_new_label[-1] = 1
            #     # print("activation no finding")
            # else:
            #     sum_topk = np.sum(topk_mat, axis=0)
            #     sum_topk[-1] = 0
            #     sum_topk[sum_topk < self.args.a2s_drop_th] = 0
            #     sum_topk[sum_topk != 0.0] = 1
            #     if sum_topk.sum() == 0.0:
            #         sum_topk[-1] = 1
            #         return sum_topk
            #     row_new_label = self.org_gt[sample_idx] * self.args.lam + sum_topk * (
            #         1 - self.args.lam
            #     )
            #     # one_array = np.ones((self.args.num_classes,))
            #     # row_new_label = (
            #     #     self.org_gt[sample_idx] * (1 - self.args.lam)
            #     #     + (self.args.lam / 2) * one_array
            #     #     + np.abs(self.args.lam) * sum_topk
            #     # )
        return row_new_label
