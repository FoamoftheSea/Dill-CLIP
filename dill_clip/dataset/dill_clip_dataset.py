import requests
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets import load_dataset


class DillCLIPValDataset(Dataset):
    def __init__(self, data_root: str, max_frames=5000):
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
