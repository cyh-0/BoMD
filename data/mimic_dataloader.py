import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

import imageio
import numpy as np
import pandas as pd
import pydicom
import skimage
import skimage.transform
from skimage.io import imread
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import xml
from loguru import logger
from PIL import Image
from skimage import io


def isNaN(string):
    return string != string


class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S.
    MIMIC-CXR: A large publicly available database of labeled chest radiographs.
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(
        self,
        imgpath,
        csvpath,
        metacsvpath,
        splitpath,
        views=["PA"],
        transform=None,
        data_aug=None,
        flat_dir=True,
        seed=0,
        unique_patients=True,
        mode=None,
    ):
        self.pathologies = [
            "Atelectasis",
            "Cardiomegaly",
            "Pleural Effusion",
            ## "Infiltration",
            ## "Mass",
            ## "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            ## "Emphysema",
            ## "Fibrosis",
            ## "Pleural_Thickening",
            ## "Hernia",
            # ---------
            "Fracture",
            "Lung Opacity",
            "Lung Lesion",
            # ---------
            "Pleural Other",
            "Enlarged Cardiomediastinum",
            # "Support Devices",
            # ---------
            "No Finding",
        ]

        # self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.splitpath = splitpath
        self.csv = pd.read_csv(self.csvpath, dtype=str)
        self.csv = self.csv.replace(np.nan, "0")
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath, dtype=str)
        self.splitpath = pd.read_csv(self.splitpath, dtype=str)

        self.metacsv = self.metacsv.set_index(["dicom_id"])
        self.splitpath = self.splitpath.set_index(["dicom_id"])
        self.metacsv = self.metacsv.join(self.splitpath.split).reset_index()

        self.csv = self.csv.set_index(["subject_id", "study_id"])
        self.metacsv = self.metacsv.set_index(["subject_id", "study_id"])

        self.csv = self.csv.join(self.metacsv).reset_index()
        if mode == "train":
            self.csv = self.csv[self.csv.split == "train"]
        elif mode == "test":
            self.csv = self.csv[self.csv.split == "test"]
        else:
            raise ValueError(f"UNKNOW MODE {mode} PLEASE CALL POLICE")

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns and pathology != "No Finding":
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            if pathology == "No Finding":
                mask = healthy.astype(np.float32)
            self.labels.append(mask.values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = 0

        # Rename pathologies
        self.pathologies = np.char.replace(
            self.pathologies, "Pleural Effusion", "Effusion"
        )

        self.trim()

        # Assign no finding
        row_sum = np.sum(self.labels, axis=1)
        self.labels[np.where(row_sum == 0), -1] = 1
        ########## add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

        logger.bind(stage="DATA").info(
            f"Num of no_finding: {(self.labels[:,-1]==1).sum()}"
        )
        logger.bind(stage="DATA").info(
            f"Maximum labels for an individual image: {np.sum(self.labels, axis=1).max()}"
        )

    def trim(self):
        # Cut label
        cut_list = [
            "Consolidation",
            "Enlarged Cardiomediastinum",
            "Pleural Other",
        ]
        for dp_class in cut_list:
            drop_idx = np.where(self.pathologies == dp_class)[0].item()
            self.pathologies = np.delete(self.pathologies, drop_idx)
            self.labels = np.delete(self.labels, drop_idx, axis=1)
        return

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        # file_name = self.csv.iloc[idx].file_name
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
        img_path = os.path.join(
            self.imgpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".jpg",
        )
        image = Image.fromarray(io.imread(img_path)).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        return image, label, idx

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if "*" in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view


def construct_mimic(args, root_dir, mode):
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

    # dataset = MIMIC_Dataset(transforms=transform)
    dataset_dir = (
        "../dataset/mimc_cxr/jpg_version/physionet.org/files/mimic-cxr-jpg/2.0.0"
    )
    dataset = MIMIC_Dataset(
        imgpath=dataset_dir + "/files",
        csvpath=dataset_dir + "/mimic-cxr-2.0.0-chexpert.csv",
        metacsvpath=dataset_dir + "/mimic-cxr-2.0.0-metadata.csv",
        splitpath=dataset_dir + "/mimic-cxr-2.0.0-split.csv",
        transform=transform,
        data_aug=None,
        unique_patients=False,
        views=["PA", "AP"],
        mode=mode,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
    )

    return loader

    # dataset = Openi_Dataset(transform)


# dataset_dir = (
#     "../dataset/mimc_cxr/jpg_version/physionet.org/files/mimic-cxr-jpg/2.0.0"
# )
# # construct_loader(args, "")
# MIMIC_Dataset(
#     imgpath="../dataset/mimc_cxr/jpg_version/physionet.org/files",
#     csvpath=dataset_dir + "/mimic-cxr-2.0.0-chexpert.csv",
#     metacsvpath=dataset_dir + "/mimic-cxr-2.0.0-metadata.csv",
#     splitpath=dataset_dir + "/mimic-cxr-2.0.0-split.csv",
#     transform=None,
#     data_aug=None,
#     unique_patients=False,
#     mode="train",
#     views=["PA", "AP"],
# )
