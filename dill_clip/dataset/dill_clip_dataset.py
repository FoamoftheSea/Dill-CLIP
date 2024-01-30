from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DillCLIPValDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = Path(data_root) / "val" if not data_root.endswith("val") else Path(data_root)
        self.frames = self._get_frames()
        with open(self.data_root / "ILSVRC2012_validation_ground_truth.txt", "r") as f:
            targets = f.read().splitlines()
            self.targets = {i + 1: int(t) for i, t in enumerate(targets)}

    def _get_frames(self):
        return sorted(list(self.data_root.glob("*.JPEG")), key=lambda x: int(x.stem.split("_")[-1]))

    def __len__(self):
        return (len(self.frames))

    def __getitem__(self, idx):
        img_path = self.frames[idx]
        lbl_path = img_path.parent / "clip_soft_labels" / f"{img_path.stem}.npy"
        img = np.array(Image.open(img_path).convert("RGB"))
        lbl = np.load(lbl_path).squeeze()
        target = self.targets[int(img_path.stem.split("_")[-1])]

        return {"pixel_values": img, "labels": lbl, "targets": target}


class DillCLIPTrainDataset(Dataset):

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.frames = self._get_frames()

    def _get_frames(self):
        with open(self.data_directory, "r") as f:
            frames = f.read().splitlines()

        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        img_path = self.frames[idx]
        lbl_path = img_path.replace("images", "clip_soft_labels").replace(".jpg", ".npy")
        img = np.array(Image.open(img_path).convert("RGB"))
        lbl = np.load(lbl_path)

        return {"pixel_values": img, "labels": lbl}
