import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VISION_FEATURE_LAYER = -2  # LLaVA authors recommend
SPLIT = "val"

img_folder = Path(f"D:/ILSVRC/Data/CLS-LOC/{SPLIT}")

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

if SPLIT == "train":
    for img_dir in tqdm(list(img_folder.iterdir())):
        labels_folder = img_dir / "clip_soft_labels"
        labels_folder.mkdir(exist_ok=True, parents=True)
        for img_path in tqdm(list(img_dir.glob("*.JPEG"))):
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt", padding=True)
            inputs["pixel_values"] = inputs.pop("pixel_values").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            target = outputs.hidden_states[VISION_FEATURE_LAYER].cpu().detach().numpy()
            out_filename = f"{img_path.stem}.npy"
            np.save(labels_folder / out_filename, target)

if SPLIT == "val":
    labels_folder = img_folder / "clip_soft_labels"
    labels_folder.mkdir(exist_ok=True, parents=True)
    for img_path in tqdm(img_folder.glob("*.JPEG")):
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs["pixel_values"] = inputs.pop("pixel_values").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        target = outputs.hidden_states[VISION_FEATURE_LAYER].cpu().detach().numpy()
        out_filename = f"{img_path.stem}.npy"
        np.save(labels_folder / out_filename, target)
