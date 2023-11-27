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
from data.padchest_dataloader_cut import PC_Dataset
import wandb
import numpy as np

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


class Chest_PDC_Dataset(Dataset):
    """
    Del sample when delete label
    """

    def __init__(
        self, root_dir, transforms, mode, file_name=None, stage="CLS", args=None
    ) -> None:
        self.stage = stage
        self.transform = transforms
        self.mode = mode
        self.args = args
        self.root_dir = root_dir
        pdc_dataset = PC_Dataset(args=args, transform=None, mode="train")
        self.load_nih()
        assert (self.pathologies[:-1] != pdc_dataset.pathologies[:-1]).sum() == 0

        self.pdc_image_path = pdc_dataset.pdc_image_path
        self.pdc_label = pdc_dataset.pdc_label
        
        np.save(os.path.join(wandb.run.dir, "pdc_label.npy"), self.pdc_label)
        self.train_img = self.nih_image_path.copy()
        self.train_img.extend(self.pdc_image_path)
        self.train_img = np.asarray(self.train_img)

        if args.mid_ckpt:
            logger.bind(stage="DATALOADER").warning("LOADING NOISY LABEL")
            self.pdc_label = np.load(f"./ckpt/saved/{args.mid_ckpt}/files/pdc_label.npy")

        self.train_label = np.concatenate((self.nih_label, self.pdc_label), axis=0)
        self.train_size = self.train_img.shape[0]
        print(f"Train size {self.train_size}")
        assert self.train_size == self.train_label.shape[0]
        del pdc_dataset

    def load_nih(
        self,
    ):
        self.root_dir_nih = os.path.join(self.root_dir, "chestxray")
        self.pathologies = np.array(list(Labels))
        self.drop_list = []
        df_path = os.path.join(self.root_dir_nih, "Data_Entry_2017.csv")
        gt = pd.read_csv(df_path, index_col=0)
        gt = gt.to_dict()["Finding Labels"]

        img_list = os.path.join(self.root_dir_nih, "train_val_list.txt")

        with open(img_list) as f:
            names = f.read().splitlines()
        self.imgs = np.asarray([x for x in names])
        gt = np.asarray([gt[i] for i in self.imgs])
        self.gt = np.zeros((gt.shape[0], 15))
        for idx, i in enumerate(gt):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.gt[idx] = binary_result

        if self.args.trim_data:
            self.trim_nih()
            # Assign no finding
            row_sum = np.sum(self.gt, axis=1)
            # self.gt[np.where(row_sum == 0), -1] = 1

            # Ensure no empty rows
            drop_idx = np.where(row_sum == 0)[0]
            self.gt = np.delete(self.gt, drop_idx, axis=0)
            self.imgs = np.delete(self.imgs, drop_idx, axis=0)

        self.nih_image_path = [
            os.path.join(self.root_dir_nih, "data", item) for item in self.imgs
        ]
        self.nih_label = self.gt.copy()

    def __getitem__(self, index):
        img_path = self.train_img[index]
        gt = self.train_label[index]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)
        if self.mode != "train" or self.stage == "MID":
            return img_t, gt, index
        else:
            return img_t, gt, index, self.knn[index]

    def __len__(self):
        return self.train_size

    def trim_nih(self):
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


def construct_cx14_pdc_cut(args, mode, file_name=None, stage="CLS"):
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
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    assert file_name is not None

    dataset = Chest_PDC_Dataset(
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
