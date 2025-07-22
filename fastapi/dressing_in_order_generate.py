import os
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import uuid
from config.settings import settings
from inference import load_model, run_inference
from utils.image_utils import save_upload_file

# Load the model - this should now use the updated paths
model = load_model()

async def generate_tryon_result(
    cloth_image_url: str,
    user_image_path: str,
    pose_image_path: str
) -> str:

    try:
        # Download cloth image
        response = requests.get(cloth_image_url)
        cloth_img = Image.open(BytesIO(response.content))
        cloth_img = np.array(cloth_img)
        
        # Load user and pose images
        user_img = cv2.imread(user_image_path)
        pose_img = cv2.imread(pose_image_path)
        
        # Run model inference
        result_img = run_inference(model, cloth_img, user_img, pose_img)
        
        # Save result
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        result_path = os.path.join(settings.RESULTS_DIR, f"result_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(result_path, result_img)
        
        return result_path
    except Exception as e:
        raise Exception(f"Error during virtual try-on: {str(e)}")