import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VISION_FEATURE_LAYER = -2  # LLaVA authors recommend
OVERWRITE = False

img_folder = Path(f"D:/yfcc100m/data/images/")
out_folder = Path(str(img_folder).replace("images", "clip_soft_labels"))

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

upr_dirs = list(img_folder.glob("2*"))
for i, upr_dir in enumerate(tqdm(upr_dirs)):
    if i < 135:
        continue
    # if upr_dir.stem.startswith("105"):
    #     continue
    # try:
    #     int(str(upr_dir.stem)[1])
    # except ValueError:
    for img_folder in tqdm(list(upr_dir.iterdir())):
        labels_folder = out_folder / upr_dir.stem / img_folder.stem
        labels_folder.mkdir(exist_ok=True, parents=True)
        for img_path in img_folder.glob("*.jpg"):
            out_filename = labels_folder / f"{img_path.stem}.npy"
            if out_filename.exists() and not OVERWRITE:
                continue
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
                inputs = processor(images=image, return_tensors="pt", padding=True)
                inputs["pixel_values"] = inputs.pop("pixel_values").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                target = outputs.hidden_states[VISION_FEATURE_LAYER][0].cpu().detach().numpy()
                np.save(out_filename, target)
            except:
                print("Skipping", img_path)
