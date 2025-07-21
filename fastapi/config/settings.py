from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    DB_NAME: str = "cloth_visually_try"
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    JWT_SECRET: str = "cloth_visually_try"
    MODEL_CHECKPOINT_PATH: str = "./dressing-in-order-main/checkpoints/latest_net_G.pth"
    STATIC_DIR: str = "static"
    TEMP_DIR: str = "static/temp"
    RESULTS_DIR: str = "static/results"
    
    class Config:
        env_file = ".env"

settings = Settings()