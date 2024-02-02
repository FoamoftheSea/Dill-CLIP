from pathlib import Path

import requests
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets import load_dataset


class DillCLIPValDataset(Dataset):
    def __init__(self, max_frames=5000):
        dataset = load_dataset("mrm8488/ImageNet1K-val", split="train")
        self.dataset = torch.utils.data.Subset(dataset, random.sample(list(range(len(dataset))), max_frames))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = np.array(item["image"].convert("RGB"))
        target = item["label"]

        return {"pixel_values": img, "targets": target}


class DillCLIPTrainDataset(Dataset):

    def __init__(self):
        dataset = load_dataset("Ziyang/yfcc15m", split="train")
        feature_names = list(dataset.features.keys())
        self.dataset = dataset.rename_columns({feature_names[0]: "url", feature_names[1]: "caption"})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        frame = self.dataset[idx]
        img_path = frame["url"]
        try:
            image = np.array(Image.open(requests.get(img_path, stream=True).raw).convert("RGB"))
            return {"pixel_values": image, "text_label": frame["caption"]}

        except Exception as e:
            print(f"Error loading frame: {img_path} \nWith error: {e}")
            return self.__getitem__(random.choice(list(range(len(self)))))


class DillCLIPLocalValDataset(Dataset):
    def __init__(self, data_root: str, max_frames=5000):
        self.data_root = Path(data_root) / "val" if not data_root.endswith("val") else Path(data_root)
        self.frames = self._get_frames()[:max_frames]
        with open(self.data_root / "ILSVRC2012_validation_ground_truth.txt", "r") as f:
            targets = f.read().splitlines()
            self.targets = {i + 1: int(t) for i, t in enumerate(targets)}

    def _get_frames(self):
        return sorted(list(self.data_root.glob("*.JPEG")), key=lambda x: int(x.stem.split("_")[-1]))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_path = self.frames[idx]
        lbl_path = img_path.parent / "clip_soft_labels" / f"{img_path.stem}.npy"
        img = np.array(Image.open(img_path).convert("RGB"))
        lbl = np.load(lbl_path).squeeze()
        target = self.targets[int(img_path.stem.split("_")[-1])]

        return {"pixel_values": img, "labels": lbl, "targets": target}


class DillCLIPLocalTrainDataset(Dataset):

    def __init__(self, data_directory: str = "train_directory.txt"):
        self.data_directory = data_directory
        self.frames = self._get_frames()

    def _get_frames(self):
        with open(self.data_directory, "r") as f:
            frames = f.read().splitlines()

        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        try:
            # img_path = self.frames[idx]
            img_path = self.frames[idx].replace("D:", "/mnt/d").replace("\\", "/")
            lbl_path = img_path.replace("images", "clip_soft_labels").replace(".jpg", ".npy")
            img = np.array(Image.open(img_path).convert("RGB"))
            lbl = np.load(lbl_path)
            return {"pixel_values": img, "labels": lbl}

        except Exception as e:
            print(f"Error loading frame: {img_path} \nWith error: {e}")
            return self.__getitem__(random.choice(list(range(len(self)))))
