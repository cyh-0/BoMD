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
    """
    Del sample when delete label
    """

    def __init__(
        self, root_dir, transforms, mode, file_name=None, stage="CLS", args=None
    ) -> None:
        self.stage = stage
        self.transform = transforms
        self.root_dir = os.path.join(root_dir, "chestxray")
        self.mode = mode
        self.pathologies = np.array(list(Labels))
        self.drop_list = []
        df_path = os.path.join(self.root_dir, "Data_Entry_2017.csv")
        self.csv = pd.read_csv(df_path, index_col=0)
        gt = self.csv.to_dict()["Finding Labels"]

        if file_name == "test":
            img_list = os.path.join(self.root_dir, "test_list.txt")
        elif file_name == "clean_test":
            self.clean_root = os.path.join(
                root_dir, "NIH_GOOGLE", "four_findings_expert_labels"
            )
            img_list = os.path.join(self.clean_root, "test_labels.csv")
        elif file_name == "train":
            img_list = os.path.join(self.root_dir, "train_val_list.txt")
        else:
            raise ValueError(f"Unknow file_name {file_name}")

        if file_name == "clean_test":
            gt = pd.read_csv(img_list, index_col=0)
            self.imgs = gt.index.to_numpy()
            gt = gt.iloc[:, -5:-1]
            # replace NO YES with 0 1
            gt = gt.replace("NO", 0)
            gt = gt.replace("YES", 1)
            clean_gt = gt.to_numpy()
            self.gt = np.zeros((clean_gt.shape[0], args.num_classes))
            # Nodule
            self.gt[:, 4] = clean_gt[:, 3]
            # Mass
            self.gt[:, 5] = clean_gt[:, 3]
            # Pheumothorax
            self.gt[:, 7] = clean_gt[:, 1]
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

            if args.trim_data:
                self.trim()
                # Assign no finding
                row_sum = np.sum(self.gt, axis=1)
                # self.gt[np.where(row_sum == 0), -1] = 1

                # Ensure no empty rows
                drop_idx = np.where(row_sum == 0)[0]
                self.gt = np.delete(self.gt, drop_idx, axis=0)
                self.imgs = np.delete(self.imgs, drop_idx, axis=0)

        self.train_img = self.imgs.copy()
        self.train_label = self.gt.copy()

        del self.imgs, self.gt

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "data", self.train_img[index])
        gt = self.train_label[index]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)
        if self.mode == "train" and self.stage == "CLS":
            return img_t, gt, index, self.knn[index]
        else:
            return img_t, gt, index

    def __len__(self):
        return self.train_img.shape[0]

    def trim(self):
        cut_list = [
            # 'Mass',
            # 'Nodule',
            "Consolidation",
        ]

        for dp_class in cut_list:
            # Find column and row related to the cut list
            drop_idx_col = np.where(self.pathologies == dp_class)[0].item()
            drop_idx_row = np.where(self.gt[:, drop_idx_col] == 1.0)[0]
            self.drop_list.append(
                {
                    "dp_class": dp_class,
                    "drop_idx_col": drop_idx_col,
                    "drop_idx_row": drop_idx_row,
                }
            )
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            # Drop gt for row and col
            self.gt = np.delete(self.gt, drop_idx_row, axis=0)
            self.gt = np.delete(self.gt, drop_idx_col, axis=1)
            # Drop imag
            self.imgs = np.delete(self.imgs, drop_idx_row, axis=0)

        return


def construct_cx14_cut(args, mode, file_name=None, stage="CLS"):
    root_dir = args.root_dir
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train" and stage == "MID":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.resize, args.resize), scale=(0.2, 1)
                ),
                transforms.ColorJitter(
                    (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)
                ),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.15, 0.15),
                    scale=(0.95, 1.05),
                    shear=(-10, 10, -10, 10),
                    fill=0,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif mode == "train" and stage == "CLS":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.resize, args.resize), scale=(0.2, 1)
                ),
                transforms.ColorJitter(
                    (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)
                ),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.15, 0.15),
                    scale=(0.95, 1.05),
                    shear=(-10, 10, -10, 10),
                    fill=0,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(args.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    assert file_name is not None

    dataset = ChestDataset(
        root_dir, transform, mode, file_name=file_name, stage=stage, args=args
    )

    shuffle = True if mode == "train" else False
    drop_last = True if mode == "train" else False

    if stage == "CLS":
        batch_size = 16
    else:
        batch_size = args.batch_size

    logger.bind(stage="DATALOADER").warning(f"Stage = [{stage}] | Mode = [{mode}]")
    logger.bind(stage="DATALOADER").warning(
        f"[Configs] | batch_size={batch_size} | shuffle={shuffle} | drop_last={drop_last}"
    )
    logger.bind(stage="DATALOADER").info(transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader

