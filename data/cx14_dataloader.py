import os

import numpy as np
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.modules import transformer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import wandb
import csv
from loguru import logger

# from utils.gcloud import download_chestxray_unzip

Labels = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
    "No Finding": 14,
}
mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestDataset(Dataset):
    def __init__(self, root_dir, transforms, mode, args=None) -> None:

        self.transform = transforms
        self.root_dir = os.path.join(root_dir, "chestxray")
        df_path = os.path.join(self.root_dir, "Data_Entry_2017.csv")
        self.mode = mode
        self.pathologies = np.array(list(Labels))
        gt = pd.read_csv(df_path, index_col=0)
        gt = gt.to_dict()["Finding Labels"]

        self.gt_knn = np.load("./relabel/run-20220124_140318-24mckzen_g2g_gt.npy")

        if mode == "test":
            img_list = os.path.join(self.root_dir, "test_list.txt")
        elif mode == "clean_test":
            self.root_dir = os.path.join(
                root_dir, "NIH_GOOGLE", "four_findings_expert_labels"
            )
            img_list = os.path.join(self.root_dir, "test_labels.csv")
        else:
            img_list = os.path.join(self.root_dir, "train_val_list.txt")

        if mode == "clean_test":
            gt = pd.read_csv(img_list, index_col=0)
            self.imgs = gt.index.to_numpy()
            gt = gt.iloc[:, -5:-1]
            # replace NO YES with 0 1
            gt = gt.replace("NO", 0)
            gt = gt.replace("YES", 1)
            clean_gt = gt.to_numpy()
            self.gt = np.zeros((clean_gt.shape[0], args.num_classes))
            # Pheumothorax
            self.gt[:, 7] = clean_gt[:, 1]
            # Nodule
            self.gt[:, 4] = clean_gt[:, 3]
            # Mass
            self.gt[:, 5] = clean_gt[:, 3]
        else:
            with open(img_list) as f:
                names = f.read().splitlines()

            self.imgs = np.asarray([x for x in names])
            gt = np.asarray([gt[i] for i in self.imgs])
            self.gt = np.zeros((gt.shape[0], 15))
            for idx, i in enumerate(gt):
                target = i.split("|")
                binary_result = mlb.fit_transform(
                    [[Labels[i] for i in target]]
                ).squeeze()
                self.gt[idx] = binary_result

            self.trim()
            # Assign no finding
            row_sum = np.sum(self.gt, axis=1)
            self.gt[np.where(row_sum == 0), -1] = 1

            tmp_col = list(self.pathologies)  # list(Labels.keys()).copy()
            # tmp_col.extend(["MeanAUC-14c"])

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "data", self.imgs[index])
        gt = self.gt[index]
        knn_gt = self.gt_knn[index]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)

        if self.mode == "train":
            return img_t, gt, index, knn_gt
        else:
            return img_t, gt, index

    def __len__(self):
        return self.imgs.shape[0]

    def trim(self):
        cut_list = [
            # 'Mass',
            # 'Nodule',
            "Consolidation",
        ]

        for dp_class in cut_list:
            drop_idx = np.where(self.pathologies == dp_class)[0].item()
            self.pathologies = np.delete(self.pathologies, drop_idx)
            self.gt = np.delete(self.gt, drop_idx, axis=1)

        return


def construct_cx14(args, root_dir, mode, file_name=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.resize, args.resize), scale=(0.2, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    else:
        transform = transforms.Compose(
            [
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dataset = ChestDataset(root_dir, transform, mode, file_name)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
    )

    return loader
