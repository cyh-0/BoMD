import collections
import torch
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

# from black import main

import imageio
import numpy as np
import pandas as pd
import pydicom
import skimage
import skimage.transform
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

nih_lables = {
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
    "No_Finding": 14,
}
cxp_labels = {
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
    "No_Finding": 13,
}
cut_list = [
    "Consolidation",
    "Enlarged Cardiomediastinum",
    "Pleural Other",
    "Support Devices",
    "Lung Opacity",
    "Lung Lesion",
]


class PC_Dataset(Dataset):
    """PadChest dataset
    Hospital San Juan de Alicante - University of Alicante

    Note that images with null labels (as opposed to normal), and images that cannot
    be properly loaded (listed as 'missing' in the code) are excluded, which makes
    the total number of available images slightly less than the total number of image
    files.
    """

    def __init__(
        self,
        imgpath=None,
        csvpath="PADCHEST_TEST_CLEAN_PA_AP.csv",
        views=["PA"],
        transform=None,
        data_aug=None,
        flat_dir=True,
        unique_patients=True,
        args=None,
    ):

        super(PC_Dataset, self).__init__()
        pc_root = args.pc_root_dir
        csvpath = os.path.join(pc_root, "PADCHEST_TEST_CLEAN_PA_AP.csv")
        imgpath = os.path.join(pc_root, "PADCHEST_TEST_CLEAN_PA_AP")

        self.pathologies = [
            "Atelectasis",
            "Consolidation",
            "Infiltration",
            "Pneumothorax",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Effusion",
            "Pneumonia",
            "Pleural_Thickening",
            "Cardiomegaly",
            "Nodule",
            "Mass",
            "Hernia",
            "Fracture",
            "Granuloma",
            "Flattened Diaphragm",
            "Bronchiectasis",
            "Aortic Elongation",
            "Scoliosis",
            "Hilar Enlargement",
            "Tuberculosis",
            "Air Trapping",
            "Costophrenic Angle Blunting",
            "Aortic Atheromatosis",
            "Hemidiaphragm Elevation",
            "Support Devices",
            "Tube'",
        ]  # the Tube' is intentional

        self.pathologies = sorted(self.pathologies)
        self.pathologies.append("No_Finding")

        mapping = dict()

        mapping["Infiltration"] = [
            "infiltrates",
            "interstitial pattern",
            "ground glass pattern",
            "reticular interstitial pattern",
            "reticulonodular interstitial pattern",
            "alveolar pattern",
            "consolidation",
            "air bronchogram",
        ]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy", "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device", "pacemaker"]
        ## the ' is to select findings which end in that word
        mapping["Tube'"] = ["stent'"]

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        self.csvpath = csvpath

        # self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)

        # Standardize view names
        self.csv.loc[
            self.csv["Projection"].isin(["AP_horizontal"]), "Projection"
        ] = "AP Supine"

        self.csv["view"] = self.csv["Projection"]
        # self.limit_to_selected_views(views)

        # Remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]

        # Remove missing files
        missing = [
            "216840111366964012819207061112010307142602253_04-014-084.png",
            "216840111366964012989926673512011074122523403_00-163-058.png",
            "216840111366964012959786098432011033083840143_00-176-115.png",
            "216840111366964012558082906712009327122220177_00-102-064.png",
            "216840111366964012339356563862009072111404053_00-043-192.png",
            "216840111366964013076187734852011291090445391_00-196-188.png",
            "216840111366964012373310883942009117084022290_00-064-025.png",
            "216840111366964012283393834152009033102258826_00-059-087.png",
            "216840111366964012373310883942009170084120009_00-097-074.png",
            "216840111366964012819207061112010315104455352_04-024-184.png",
            "216840111366964012819207061112010306085429121_04-020-102.png",
        ]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        # Get our classes.
        self.gt = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            self.gt.append(mask.values)
        self.gt = np.asarray(self.gt).T
        self.gt = self.gt.astype(np.float32)

        self.pathologies[self.pathologies.index("Tube'")] = "Tube"

        ########## add consistent csv values

        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.view(int) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        self.gt[np.where(np.sum(self.gt, axis=1) == 0), -1] = 1
        if args.trim_data:
            self.trim(args.train_data)
            row_sum = np.sum(self.gt, axis=1)
            # self.gt[np.where(row_sum == 0), -1] = 1
            drop_idx = np.where(row_sum == 0)[0]
            if len(drop_idx) != 0:
                self.gt = np.delete(self.gt, drop_idx, axis=0)
                # self.imgs = np.delete(self.imgs, drop_idx, axis=0)
                self.csv = self.csv.drop(self.csv.iloc(drop_idx).index)

        if args.train_data == "NIH":
            # Permutation
            nih_train_list = list(nih_lables)
            nih_train_list.remove("Consolidation")
            order = [
                np.where(item == self.pathologies)[0].item() for item in nih_train_list
            ]
            self.pathologies = self.pathologies[order]
            self.gt = self.gt[:, order]
        else:
            cxp_train_list = list(cxp_labels)
            for i in cut_list:
                cxp_train_list.remove(i)
            order = [
                np.where(item == self.pathologies)[0].item() for item in cxp_train_list
            ]
            self.pathologies = self.pathologies[order]
            self.gt = self.gt[:, order]

    def trim(self, train_data):
        self.pathologies = np.array(self.pathologies)
        # Cut label
        NIH_CUT_LIST = sorted(list(set(self.pathologies) - set(nih_lables)))
        NIH_CUT_LIST.append("Consolidation")

        MIMIC_CUT_LIST = sorted(list(set(self.pathologies) - set(cxp_labels)))
        MIMIC_CUT_LIST.append("Consolidation")
        MIMIC_CUT_LIST.append("Support Devices")

        if train_data == "CXP":
            cut_list = MIMIC_CUT_LIST
        elif train_data == "NIH":
            cut_list = NIH_CUT_LIST
        else:
            raise ValueError(f"TRAIN DATA {train_data}")

        for dp_class in cut_list:
            drop_idx_col = np.where(self.pathologies == dp_class)[0].item()
            drop_idx_row = np.where(self.gt[:, drop_idx_col] == 1.0)[0]
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                self.pathologies = np.delete(self.pathologies, drop_idx_col)
                self.gt = np.delete(self.gt, drop_idx_col, axis=1)
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            self.gt = np.delete(self.gt, drop_idx_row, axis=0)
            self.gt = np.delete(self.gt, drop_idx_col, axis=1)
            self.csv = self.csv.drop(self.csv.iloc[drop_idx_row].index)
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
            # print(np.nonzero(self.gt[:, i] == 1)[0].shape[0])

        tmp.append(no_finding_prob)

        return tmp

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(
            len(self), self.views, self.data_aug
        )

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        imgid = self.csv["ImageID"].iloc[index]
        img_path = os.path.join(self.imgpath, imgid)
        gt = self.gt[index]
        # img = io.imread(img_path)
        img = Image.open(img_path)
        # table = [i / 256 for i in range(65536)]
        # img = img.point(table, "L")

        img = img.convert("RGB")
        img_t = self.transform(img)
        return img_t, gt, index

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.gt[idx]

        imgid = self.csv["ImageID"].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        exit()
        sample["img"] = normalize(img, maxval=65535, reshape=True)

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return sample


def construct_pc_cut(args, mode, file_name=None):
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

    dataset = PC_Dataset(args=args, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
    )

    prior = dataset.calc_prior()
    # print(prior)

    return loader, torch.from_numpy(np.asarray(prior)).to(0)


# if __name__ == "__main__":
#     PC_Dataset(imgpath="../dataset/PADCHEST/images/")
