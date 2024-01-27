import os

from pathlib import Path

data_roots = ["D:/yfcc100m/data/", "E:/yfcc100m/data/"]
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
        paths.extend(return_paths(wi, "npy"))
        if len(paths) > 1000:
            paths = dump_paths(paths)

with open(output_dir / "train_labels.txt", "w") as f:
    f.writelines(list(paths))
