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
    def __init__(self, root_dir, transforms, mode, args) -> None:
        super(CXPDataset, self).__init__()
        self.transform = transforms
        self.root_dir = root_dir
        self.mode = mode
        self.pathologies = np.asarray(list(Labels))

        df_path = os.path.join(root_dir, f"filtered_{mode}.csv")
        df = pd.read_csv(df_path, index_col=1)
        # filter Lateral

        self.all_gt = df.iloc[:, 5:].to_numpy()
        self.all_imgs = df.index.to_numpy()
        self.all_imgs = np.array(list(map(lambda x: "/".join((x.split("/")[1:])), self.all_imgs)))

        if args.trim_data:
            self.trim()
            # assign no finding
            row_sum = np.sum(self.all_gt, axis=1)
            drop_idx = np.where(row_sum == 0)[0]
            self.all_gt = np.delete(self.all_gt, drop_idx, axis=0)
            self.all_imgs = np.delete(self.all_imgs, drop_idx, axis=0)

        self.label_distribution = self.calc_prior()

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        gt = self.all_gt[index]
        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)

        return img_t, gt, index

    def __len__(self):
        return self.all_imgs.shape[0]

    def calc_prior(self):
        tmp = []
        no_finding_prob = np.nonzero(self.all_gt[:, -1] == 1)[0].shape[0] / self.all_gt.shape[0]
        for i in range(self.all_gt.shape[1] - 1):
            tmp.append(
                (1 - no_finding_prob)
                * np.nonzero(self.all_gt[:, i] == 1)[0].shape[0]
                / self.all_gt.shape[0]
            )
        tmp.append(no_finding_prob)
        return tmp

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
            drop_idx_row = np.where(self.all_gt[:, drop_idx_col] == 1.0)[0]
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            self.all_gt = np.delete(self.all_gt, drop_idx_row, axis=0)
            self.all_imgs = np.delete(self.all_imgs, drop_idx_row, axis=0)

            self.all_gt = np.delete(self.all_gt, drop_idx_col, axis=1)


def construct_cxp_cut(args, root_dir, mode):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train" or mode == "influence":
        transform = transforms.Compose(
            [
                # transforms.Resize((args.resize, args.resize)),
                # CutoutPIL(0.5),
                transforms.RandomResizedCrop((args.resize, args.resize), scale=(0.2, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-10, 10)),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
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

    dataset = CXPDataset(root_dir, transform, mode, args)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size if mode != "influence" else 1,
        shuffle=True if mode == "train" else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader, dataset.label_distribution
