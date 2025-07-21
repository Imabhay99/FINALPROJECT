import os
import uuid
import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile
from config.settings import settings

async def save_upload_file(upload_file: UploadFile, prefix: str = "", directory: str = None) -> str:
    try:
        if directory is None:
            directory = settings.TEMP_DIR

        os.makedirs(directory, exist_ok=True)
        extension = os.path.splitext(upload_file.filename)[1]
        filename = f"{prefix}{uuid.uuid4().hex}{extension}"
        file_path = os.path.join(directory, filename)

        with open(file_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)

        return file_path
    except Exception as e:
        raise Exception(f"Failed to save upload file: {str(e)}")

async def process_image(upload_file: UploadFile) -> np.ndarray:
    contents = await upload_file.read()
    image = Image.open(io.BytesIO(contents))

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = np.array(image)
    image = resize_and_crop(image, target_size=(512, 384))
    image = normalize_image(image)
    return image

def resize_and_crop(image: np.ndarray, target_size: tuple) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect = w / h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        start = (w - new_w) // 2
        image = image[:, start:start + new_w]
    else:
        new_h = int(w / target_aspect)
        start = (h - new_h) // 2
        image = image[start:start + new_h, :]

    return cv2.resize(image, target_size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 127.5) - 1.0

def denormalize_image(image: np.ndarray) -> np.ndarray:
    return ((image + 1.0) * 127.5).astype(np.uint8)

def encode_image(image: np.ndarray) -> str:
    image = denormalize_image(image)
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")