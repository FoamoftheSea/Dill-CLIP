"""
Code copied from https://raw.githubusercontent.com/parallel-domain/pd-sdk/main/paralleldomain/utilities/fsio.py
"""
from io import BytesIO

import cv2
import numpy as np
from dill_clip.utilities.any_path import AnyPath
from PIL import Image


def read_image(path: AnyPath, convert_to_rgb: bool = True, is_indexed=False) -> np.ndarray:
    with path.open(mode="rb") as fp:
        if is_indexed:
            pil_image = Image.open(BytesIO(fp.read()))
            return np.asarray(pil_image)
        else:
            return read_image_bytes(images_bytes=fp.read(), convert_to_rgb=convert_to_rgb)


def read_image_bytes(images_bytes: bytes, convert_to_rgb: bool = True) -> np.ndarray:
    image_data = cv2.imdecode(
        buf=np.frombuffer(images_bytes, np.uint8),
        flags=cv2.IMREAD_UNCHANGED,
    )
    if convert_to_rgb:
        color_convert_code = cv2.COLOR_BGR2RGB
        if image_data.shape[-1] == 4:
            color_convert_code = cv2.COLOR_BGRA2RGBA

        image_data = cv2.cvtColor(
            src=image_data,
            code=color_convert_code,
        )
    return image_data
