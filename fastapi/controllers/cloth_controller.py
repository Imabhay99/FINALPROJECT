
import os
import uuid
import cloudinary.uploader
from datetime import datetime
from fastapi import UploadFile, HTTPException
from config.settings import settings
from motor.motor_asyncio import AsyncIOMotorClient
from dressing_in_order_generate import generate_tryon_result
from utils.image_utils import save_upload_file

# MongoDB setup
client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
clothes_collection = db["clothes"]
user_photos_collection = db["user_photos"]
tryon_results_collection = db["tryon_results"]

async def list_clothes(skip: int = 0, limit: int = 10):
    clothes = await clothes_collection.find().skip(skip).limit(limit).to_list(limit)
    return clothes

async def upload_cloth_image(
    name: str,
    category: str,
    description: str,
    image: UploadFile
):
    try:
        # Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(
            image.file,
            folder="virtual_tryon/clothes",
            public_id=f"cloth_{uuid.uuid4().hex}"
        )
        
        # Save cloth metadata to MongoDB
        cloth_data = {
            "_id": str(uuid.uuid4()),
            "name": name,
            "category": category,
            "description": description,
            "image_url": upload_result["secure_url"],
            "public_id": upload_result["public_id"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await clothes_collection.insert_one(cloth_data)
        if result.inserted_id:
            return {"message": "Cloth uploaded successfully", "cloth_id": result.inserted_id}
        raise HTTPException(status_code=500, detail="Failed to upload cloth")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def upload_user_photo(
    user_image: UploadFile,
    pose_image: UploadFile
):
    try:
        # Save user and pose images to temporary files
        user_image_path = await save_upload_file(user_image, "user_", settings.TEMP_DIR)
        pose_image_path = await save_upload_file(pose_image, "pose_", settings.TEMP_DIR)
        
        # Store in DB and return IDs
        user_photo_id = str(uuid.uuid4())
        await user_photos_collection.insert_one({
            "_id": user_photo_id,
            "user_image_path": user_image_path,
            "pose_image_path": pose_image_path,
            "created_at": datetime.utcnow()
        })
        
        return {"user_photo_id": user_photo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_tryon(
    cloth_id: str,
    user_photo_id: str
):
    try:
        # Get cloth and user photos
        cloth = await clothes_collection.find_one({"_id": cloth_id})
        user_photos = await user_photos_collection.find_one({"_id": user_photo_id})
        
        if not cloth or not user_photos:
            raise HTTPException(status_code=404, detail="Cloth or user photos not found")
        
        # Generate try-on result
        result_image_path = await generate_tryon_result(
            cloth_image_url=cloth["image_url"],
            user_image_path=user_photos["user_image_path"],
            pose_image_path=user_photos["pose_image_path"]
        )
        
        # Upload result to Cloudinary
        result = cloudinary.uploader.upload(
            result_image_path,
            folder="virtual_tryon/results",
            public_id=f"result_{uuid.uuid4().hex}"
        )
        
        # Store result in DB
        tryon_id = str(uuid.uuid4())
        await tryon_results_collection.insert_one({
            "_id": tryon_id,
            "cloth_id": cloth_id,
            "user_photo_id": user_photo_id,
            "result_url": result["secure_url"],
            "created_at": datetime.utcnow(),
            "status": "completed"
        })
        
        return {"tryon_id": tryon_id, "status": "processing completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_tryon_result(tryon_id: str):
    result = await tryon_results_collection.find_one({"_id": tryon_id})
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result









# import os
# import sys
# import cloudinary
# from dressing_in_order.models.dior_model import DiorModel
# from dressing_in_order.utils.util import save_image

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# cloudinary.config(
#     cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME', 'drxs4yu5j'),
#     api_key=os.getenv('CLOUDINARY_API_KEY', '792394351774396'),
#     api_secret=os.getenv('CLOUDINARY_API_SECRET', 'FlYEHsvl6g_sKNSm8P46hf1GdYg')
# )

# model = DiorModel()

# def process_try_on(user_image_url, cloth_image_url):
#     user_temp_path = 'temp_user.jpg'
#     cloth_temp_path = 'temp_cloth.jpg'
#     save_image(user_image_url, user_temp_path)
#     save_image(cloth_image_url, cloth_temp_path)
#     result_image = model.virtual_try_on(user_temp_path, cloth_temp_path)
#     result_upload = cloudinary.uploader.upload(result_image, folder='try_on/results')
#     result_url = result_upload['secure_url']
#     os.remove(user_temp_path)
#     os.remove(cloth_temp_path)
#     os.remove(result_image)
#     return result_url










# # controllers/cloth_controller.py
# import os
# import uuid
# import subprocess
# from fastapi import UploadFile
# from fastapi.responses import JSONResponse

# UPLOAD_DIR = "uploads/user_images"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# async def process_virtual_tryon(image: UploadFile, prompt: str):
#     try:
#         # Save image
#         ext = os.path.splitext(image.filename)[-1]
#         filename = f"{uuid.uuid4()}{ext}"
#         filepath = os.path.join(UPLOAD_DIR, filename)

#         with open(filepath, "wb") as f:
#             f.write(await image.read())

#         # Move to correct data location or simulate VITON-compatible input
#         # Here, you should copy or generate correct input files your model needs
#         # For now, we assume your model is already using `data/` directory.

#         # Run the existing model pipeline (generate_all.py)
#         cmd = [
#             "python", "generate_all.py",
#             "--model", "adgan",
#             "--gpu_ids", "-1"
#         ]
#         process = subprocess.run(cmd, capture_output=True, text=True)

#         if process.returncode != 0:
#             return JSONResponse(status_code=500, content={
#                 "error": "Model execution failed",
#                 "details": process.stderr
#             })

#         return {
#             "message": "Virtual try-on success",
#             "logs": process.stdout,
#             "input_image": filename,
#             "prompt_used": prompt,
#         }
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
# controllers/cloth_controller.py
# import os
# import uuid
# import subprocess
# from fastapi import UploadFile
# from fastapi.responses import JSONResponse

# UPLOAD_DIR = "uploads/user_images"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# async def process_virtual_tryon(image: UploadFile, prompt: str):
#     try:
#         # Save image
#         ext = os.path.splitext(image.filename)[-1]
#         filename = f"{uuid.uuid4()}{ext}"
#         filepath = os.path.join(UPLOAD_DIR, filename)

#         with open(filepath, "wb") as f:
#             f.write(await image.read())

#         # Move to correct data location or simulate VITON-compatible input
#         # Here, you should copy or generate correct input files your model needs
#         # For now, we assume your model is already using `data/` directory.

#         # Run the existing model pipeline (generate_all.py)
#         cmd = [
#             "python", "generate_all.py",
#             "--model", "adgan",
#             "--gpu_ids", "-1"
#         ]
#         process = subprocess.run(cmd, capture_output=True, text=True)

#         if process.returncode != 0:
#             return JSONResponse(status_code=500, content={
#                 "error": "Model execution failed",
#                 "details": process.stderr
#             })

#         return {
#             "message": "Virtual try-on success",
#             "logs": process.stdout,
#             "input_image": filename,
#             "prompt_used": prompt,
#         }
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

