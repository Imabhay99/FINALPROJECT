from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings(BaseSettings):
    # Basic App Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    WORKERS: int = 1

    # Database
    MONGO_URI: str
    DB_NAME: str = "cloth_visually_try"

    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

    # JWT
    JWT_SECRET: str = "cloth_visually_try"

    # Model Checkpoints
    MODEL_CHECKPOINT_PATH: str = "./dressing_in_order_main/checkpoints/latest_net_G.pth"
    CLOTH_MODEL_PATH: Optional[str] = None
    SEGMENTATION_MODEL_PATH: Optional[str] = None

    # Static and File Settings
    STATIC_DIR: str = "static"
    TEMP_DIR: str = "static/temp"
    RESULTS_DIR: str = "static/results"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Asset Paths
    CLOTH_DIR: str = "./assets/cloth"
    CLOTH_MASK_DIR: str = "./assets/cloth-mask"
    PERSON_DIR: str = "./assets/person"
    PERSON_MASK_DIR: str = "./assets/person-mask"
    OUTPUT_DIR: str = "./assets/output"

    # Runtime Device
    DEVICE: str = "cuda" if os.getenv("USE_CUDA") == "1" else "cpu"

    # Output Configuration
    UPSAMPLE_OUTPUT: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
