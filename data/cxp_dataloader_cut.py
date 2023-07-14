from operator import index
import random
import os
from random import random
import numpy as np
import pandas as pd
from torch import classes

import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

Labels = {
    "Enlarged Cardiomediastinum": 0,
    "Cardiomegaly": 1,
    "Lung Opacity": 2,
    "Lung Lesion": 3,
    "Edema": 4,
    "Consolidation": 5,
    "Pneumonia": 6,
    "Atelectasis": 7,
    "Pneumothorax": 8,
    "Effusion": 9,
    "Pleural Other": 10,
    "Fracture": 11,
    "Support Devices": 12,
    "No Finding": 13,
}
mlb = MultiLabelBinarizer(classes=list(range(14)))


class CXPDataset(Dataset):
    def __init__(
        self, root_dir, transforms, mode, file_name=None, stage="CLS", args=None
    ) -> None:
        super(CXPDataset, self).__init__()
        self.stage = stage
        self.transform = transforms
        self.root_dir = os.path.join(root_dir, "CheXpert-v1.0-small")
        self.mode = mode
        self.pathologies = np.asarray(list(Labels))

        df_path = os.path.join(self.root_dir, f"filtered_{file_name}.csv")
        self.df = pd.read_csv(df_path, index_col=1)
        # filter Lateral

        self.gt = self.df.iloc[:, 5:].to_numpy()
        self.all_imgs = self.df.index.to_numpy()
        self.all_imgs = np.array(
            list(map(lambda x: "/".join((x.split("/")[1:])), self.all_imgs))
        )

        if args.trim_data:
            self.trim()
            # assign no finding
            row_sum = np.sum(self.gt, axis=1)
            drop_idx = np.where(row_sum == 0)[0]
            self.gt = np.delete(self.gt, drop_idx, axis=0)
            self.all_imgs = np.delete(self.all_imgs, drop_idx, axis=0)

            # self.df = self.df.drop(self.df.iloc[drop_idx].index)

        self.train_img = self.all_imgs.copy()
        self.train_label = self.gt.copy()

        del self.all_imgs, self.gt

        # self.label_distribution = self.calc_prior()

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.train_img[index])
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
            "Consolidation",
            "Enlarged Cardiomediastinum",
            "Pleural Other",
            "Support Devices",
            "Lung Opacity",
            "Lung Lesion",
        ]

        for dp_class in cut_list:
            drop_idx_col = np.where(self.pathologies == dp_class)[0].item()
            drop_idx_row = np.where(self.gt[:, drop_idx_col] == 1.0)[0]
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            self.gt = np.delete(self.gt, drop_idx_row, axis=0)
            self.all_imgs = np.delete(self.all_imgs, drop_idx_row, axis=0)
            self.gt = np.delete(self.gt, drop_idx_col, axis=1)
            # self.df = self.df.drop(self.df.iloc[drop_idx_row].index)
        return

    def calc_prior(self):
        tmp = []
        no_finding_prob = np.nonzero(self.gt[:, -1] == 1)[0].shape[0] / self.gt.shape[0]
        for i in range(self.gt.shape[1] - 1):
            tmp.append(
                (1 - no_finding_prob)
                * np.nonzero(self.gt[:, i] == 1)[0].shape[0]
                / self.gt.shape[0]
            )
        tmp.append(no_finding_prob)
        return tmp


def construct_cxp_cut(args, mode, file_name=None, stage="CLS"):
    root_dir = args.root_dir
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train" and stage == "VAL":
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
    dataset = CXPDataset(
        root_dir, transform, mode, file_name=file_name, stage=stage, args=args
    )

    shuffle = True if mode == "train" else False
    drop_last = True if mode == "train" else False
    
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
