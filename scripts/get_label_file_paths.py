import os

from pathlib import Path
from PIL import Image

data_roots = [
    "D:/yfcc100m/data/",
    # "E:/yfcc100m/data/",
]
output_dir = Path("./")


def return_paths(walk_item, ext):
    return [str(Path(walk_item[0]) / p) for p in walk_item[2] if p.endswith(ext)]


def dump_paths(paths):
    with open(output_dir / "train_directory.txt", "a") as f:
        f.writelines([p + "\n" for p in paths])

    paths = []
    return paths

paths = []

path_futures = set()
for data_root in data_roots:
    labels_path = Path(data_root) / "clip_soft_labels"
    for wi in os.walk(labels_path):
        npy_paths = return_paths(wi, "npy")
        valid_paths = []
        if len(npy_paths) > 0:
            for npy_path in npy_paths:
                img_path = str(npy_path).replace("clip_soft_labels", "images").replace(".npy", ".jpg")
                try:
                    im = Image.open(img_path).convert("RGB")
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Invalid image path: {img_path} \n With error:\n {e}")

        paths.extend(valid_paths)
        if len(paths) > 1000:
            paths = dump_paths(paths)
