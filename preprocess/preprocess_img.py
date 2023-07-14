import collections
import os
from pickletools import read_unicodestring1
import numpy as np
import pandas as pd
import skimage.transform
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
import torchvision.transforms.functional as TF
from tqdm import tqdm

root_dir = "../dataset/PADCHEST"
files = os.listdir("../dataset/PADCHEST/pc_test_images")


def tmpFunc(img_path):
    img = Image.open(img_path)
    table = [i / 256 for i in range(65536)]
    img = img.point(table, "L")
    img = TF.resize(img, (512, 512))
    name = img_path.split("/")[-1]
    img.save(os.path.join(root_dir, "pc_test_images_resize/{}".format(name)))
    return


def applyParallel(func):
    with Pool(cpu_count()) as p:
        p.map(
            func,
            [
                os.path.join(root_dir, "pc_test_images", files[i])
                for i in tqdm(range(len(files)))
            ],
        )


if __name__ == "__main__":
    processed_df = applyParallel(tmpFunc)
