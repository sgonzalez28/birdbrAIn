"""
train_trait_model.py

Train a visual classifier for a single morphological trait.

This script will:
- Merge metadata
- Builds dataloaders
- Applies Albumentations augmentation
- Performs transfer learning with ResNet34
- Saves frozen and unfrozen models

Example:
    python train_trait_model.py --config configs/train_config.yaml
"""

# -------------------------
# 1. Imports. 
# These are the packages used to run the training for the single morphological trait.
# -------------------------

import argparse
import os
import pandas as pd
from pathlib import Path
import yaml

from fastai.vision.all import *
from fastai.metrics import BalancedAccuracy, F1Score
import albumentations
import numpy as np


# -------------------------
# 2. Configuration
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# 3. Data Preparation
# -------------------------

def merge_metadata(df0_path, df1_path, trait_column, image_dir):
    """
    Merge metadata from two species sources
    and prune to match available images.
    """

    df0 = pd.read_csv(df0_path)
    df1 = pd.read_csv(df1_path)

    df0['SP'] = 0
    df1['SP'] = 1

    df0 = df0[['ID', trait_column]].copy()
    df1 = df1[['ID', trait_column]].copy()

    df = pd.concat([df0, df1])

    # Prune to match image files
    file_list = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]
    df = df[df['ID'].isin(file_list)].reset_index(drop=True)

    df.rename(columns={'ID': 'name', trait_column: 'label'}, inplace=True)

    return df


# -------------------------
# 4. Data Augmentation
# -------------------------

def get_train_aug():
    return albumentations.Compose([
        albumentations.RandomResizedCrop(300, 300),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(p=0.5),
        albumentations.CoarseDropout(p=0.5),
    ])


class AlbumentationsTransform(DisplayedTransform):
    split_idx, order = 0, 2

    def __init__(self, aug):
        self.aug = aug

    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


# -------------------------
# 5. Training Function
# -------------------------

def train(config):

    image_dir = Path(config["image_dir"])
    df = merge_metadata(
        config["df0_path"],
        config["df1_path"],
        config["trait_column"],
        image_dir
    )

    item_tfms = [
        Resize(config["image_size"]),
        AlbumentationsTransform(get_train_aug())
    ]

    dls = ImageDataLoaders.from_df(
        df,
        path=image_dir,
        suff=config["image_suffix"],
        valid_pct=config["valid_pct"],
        item_tfms=item_tfms
    )

    learn = vision_learner(
        dls,
        resnet34,
        pretrained=True,
        metrics=[error_rate, BalancedAccuracy(), F1Score()]
    )

    # ---- Stage 1: Frozen ----
    lr = learn.lr_find().valley
    learn.fit_one_cycle(config["epochs_stage1"], slice(lr))
    learn.save(config["save_name_stage1"])

    # ---- Stage 2: Unfrozen ----
    learn.unfreeze()
    lr = learn.lr_find().valley
    learn.fit_one_cycle(
        config["epochs_stage2"],
        lr_max=slice(lr, lr/5)
    )
    learn.save(config["save_name_stage2"])

    print("Training complete.")


# -------------------------
# 6. Main
# -------------------------

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)