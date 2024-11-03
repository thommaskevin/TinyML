import os.path

import numpy as np
from os import listdir
from glob import glob
from PIL import Image


def load_folder(folder: str):
    """
    Load images from folder as [0, 1] floats
    :param folder:
    :return:
    """
    for filename in sorted(glob(f"{folder}/*.jpg") + glob(f"{folder}/*.jpeg")):
        yield np.asarray(Image.open(filename).convert("L"), dtype=float) / 255.


def load_split(root: str, split_name: str):
    """
    Load images from train/val/test folder
    :param root:
    :param split_name:
    :return:
    """
    X = []
    Y = []
    folders = listdir(f"{root}/{split_name}")
    folders = [f"{root}/{split_name}/{f}" for f in folders if os.path.isdir(f"{root}/{split_name}/{f}")]

    for k, folder in enumerate(sorted(folders)):
        folder_x = list(load_folder(folder))
        X += folder_x
        Y += [k] * len(folder_x)

    # shuffle inputs
    shuffle_mask = np.random.permutation(len(X))
    X = np.asarray(X)[shuffle_mask]
    Y = np.asarray(Y)[shuffle_mask]

    return X, Y
