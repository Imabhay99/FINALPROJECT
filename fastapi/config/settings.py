from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MONGO_URI: str
    DEBUG: bool = True
    WORKERS: int = 1
    DB_NAME: str = "cloth_visually_try"
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    JWT_SECRET: str = "cloth_visually_try"
    MODEL_CHECKPOINT_PATH: str = "./dressing-in-order-main/checkpoints/latest_net_G.pth"
    STATIC_DIR: str = "static"
    TEMP_DIR: str = "static/temp"
    RESULTS_DIR: str = "static/results"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    CLOTH_MODEL_PATH: str
    SEGMENTATION_MODEL_PATH: str
    DEVICE: str = "cuda" if os.environ.get("USE_CUDA") else "cpu"
    CLOTH_DIR: str = "./assets/cloth"
    CLOTH_MASK_DIR: str = "./assets/cloth-mask"
    PERSON_DIR: str = "./assets/person"
    PERSON_MASK_DIR: str = "./assets/person-mask"
    OUTPUT_DIR: str = "./assets/output"
    UPSAMPLE_OUTPUT: bool = True
    CLOTH_MODEL_PATH: Optional[str] = None
    SEGMENTATION_MODEL_PATH: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()
