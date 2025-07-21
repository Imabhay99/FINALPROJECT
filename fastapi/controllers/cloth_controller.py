# controllers/cloth_controller.py
import uuid
import cloudinary.uploader
from datetime import datetime
from fastapi import UploadFile, HTTPException
from config.settings import settings
from motor.motor_asyncio import AsyncIOMotorClient
from dressing_in_order_generate import generate_tryon_result
from utils import image_utils, model_utils
import numpy as np
import logging

logger = logging.getLogger(__name__)

client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
clothes_collection = db["clothes"]
user_photos_collection = db["user_photos"]
tryon_results_collection = db["tryon_results"]

async def list_clothes(skip: int = 0, limit: int = 10):
    return await clothes_collection.find().skip(skip).limit(limit).to_list(length=limit)

async def upload_cloth_image(name: str, category: str, description: str, image: UploadFile):
    try:
        upload_result = cloudinary.uploader.upload(
            image.file,
            folder="virtual_tryon/clothes",
            public_id=f"cloth_{uuid.uuid4().hex}"
        )
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
        return {"message": "Cloth uploaded successfully", "cloth_id": cloth_data["_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def upload_user_photo(user_image: UploadFile, pose_image: UploadFile):
    try:
        user_path = await image_utils.save_upload_file(user_image, "user_", settings.TEMP_DIR)
        pose_path = await image_utils.save_upload_file(pose_image, "pose_", settings.TEMP_DIR)
        user_photo_id = str(uuid.uuid4())
        await user_photos_collection.insert_one({
            "_id": user_photo_id,
            "user_image_path": user_path,
            "pose_image_path": pose_path,
            "created_at": datetime.utcnow()
        })
        return {"user_photo_id": user_photo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_tryon(cloth_id: str, user_photo_id: str):
    try:
        cloth = await clothes_collection.find_one({"_id": cloth_id})
        user_photos = await user_photos_collection.find_one({"_id": user_photo_id})

        if not cloth or not user_photos:
            raise HTTPException(status_code=404, detail="Cloth or user photo not found")

        result_image_path = await generate_tryon_result(
            cloth_image_url=cloth["image_url"],
            user_image_path=user_photos["user_image_path"],
            pose_image_path=user_photos["pose_image_path"]
        )

        result = cloudinary.uploader.upload(
            result_image_path,
            folder="virtual_tryon/results",
            public_id=f"result_{uuid.uuid4().hex}"
        )

        tryon_id = str(uuid.uuid4())
        await tryon_results_collection.insert_one({
            "_id": tryon_id,
            "cloth_id": cloth_id,
            "user_photo_id": user_photo_id,
            "result_url": result["secure_url"],
            "created_at": datetime.utcnow(),
            "status": "completed"
        })
        return {"tryon_id": tryon_id, "result_url": result["secure_url"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_tryon_result(tryon_id: str):
    result = await tryon_results_collection.find_one({"_id": tryon_id})
    if not result:
        raise HTTPException(status_code=404, detail="Try-on result not found")
    return result

model = model_utils.load_dior_model()

class ClothController:
    def __init__(self):
        self.model = model_utils.load_dior_model()

    async def generate_outfit(self, person_img: UploadFile, cloth_img: UploadFile, pose_data: dict = None):
        try:
            person = await image_utils.process_image(person_img)
            cloth = await image_utils.process_image(cloth_img)
            result = self._run_inference(person, cloth, pose_data)
            return image_utils.encode_image(result)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise HTTPException(500, "Outfit generation failed")

    def _run_inference(self, person: np.ndarray, cloth: np.ndarray, pose_data: dict = None):
        model_input = {
            "person": person,
            "cloth": cloth,
            "pose": pose_data
        }
        return self.model.generate(model_input)

cloth_controller = ClothController()












# import os
# import uuid
# import cloudinary.uploader
# from datetime import datetime
# from fastapi import UploadFile, HTTPException, APIRouter
# from config.settings import settings
# from motor.motor_asyncio import AsyncIOMotorClient
# from dressing_in_order_generate import generate_tryon_result
# from utils.image_utils import save_upload_file
# from utils import image_utils, model_utils
# from fastapi.exceptions import HTTPException
# from ..utils import image_utils, model_utils
# import numpy as np
# import logging

# logger = logging.getLogger(__name__)

# # MongoDB setup
# client = AsyncIOMotorClient(settings.MONGO_URI)
# db = client[settings.DB_NAME]
# clothes_collection = db["clothes"]
# user_photos_collection = db["user_photos"]
# tryon_results_collection = db["tryon_results"]

# async def list_clothes(skip: int = 0, limit: int = 10):
#     clothes = await clothes_collection.find().skip(skip).limit(limit).to_list(limit)
#     return clothes

# async def upload_cloth_image(
#     name: str,
#     category: str,
#     description: str,
#     image: UploadFile
# ):
#     try:
#         # Upload image to Cloudinary
#         upload_result = cloudinary.uploader.upload(
#             image.file,
#             folder="virtual_tryon/clothes",
#             public_id=f"cloth_{uuid.uuid4().hex}"
#         )
        
#         # Save cloth metadata to MongoDB
#         cloth_data = {
#             "_id": str(uuid.uuid4()),
#             "name": name,
#             "category": category,
#             "description": description,
#             "image_url": upload_result["secure_url"],
#             "public_id": upload_result["public_id"],
#             "created_at": datetime.utcnow(),
#             "updated_at": datetime.utcnow()
#         }
        
#         result = await clothes_collection.insert_one(cloth_data)
#         if result.inserted_id:
#             return {"message": "Cloth uploaded successfully", "cloth_id": result.inserted_id}
#         raise HTTPException(status_code=500, detail="Failed to upload cloth")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# async def upload_user_photo(
#     user_image: UploadFile,
#     pose_image: UploadFile
# ):
#     try:
#         # Save user and pose images to temporary files
#         user_image_path = await save_upload_file(user_image, "user_", settings.TEMP_DIR)
#         pose_image_path = await save_upload_file(pose_image, "pose_", settings.TEMP_DIR)
        
#         # Store in DB and return IDs
#         user_photo_id = str(uuid.uuid4())
#         await user_photos_collection.insert_one({
#             "_id": user_photo_id,
#             "user_image_path": user_image_path,
#             "pose_image_path": pose_image_path,
#             "created_at": datetime.utcnow()
#         })
        
#         return {"user_photo_id": user_photo_id}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# async def generate_tryon(
#     cloth_id: str,
#     user_photo_id: str
# ):
#     try:
#         # Get cloth and user photos
#         cloth = await clothes_collection.find_one({"_id": cloth_id})
#         user_photos = await user_photos_collection.find_one({"_id": user_photo_id})
        
#         if not cloth or not user_photos:
#             raise HTTPException(status_code=404, detail="Cloth or user photos not found")
        
#         # Generate try-on result
#         result_image_path = await generate_tryon_result(
#             cloth_image_url=cloth["image_url"],
#             user_image_path=user_photos["user_image_path"],
#             pose_image_path=user_photos["pose_image_path"]
#         )
        
#         # Upload result to Cloudinary
#         result = cloudinary.uploader.upload(
#             result_image_path,
#             folder="virtual_tryon/results",
#             public_id=f"result_{uuid.uuid4().hex}"
#         )
        
#         # Store result in DB
#         tryon_id = str(uuid.uuid4())
#         await tryon_results_collection.insert_one({
#             "_id": tryon_id,
#             "cloth_id": cloth_id,
#             "user_photo_id": user_photo_id,
#             "result_url": result["secure_url"],
#             "created_at": datetime.utcnow(),
#             "status": "completed"
#         })
        
#         return {"tryon_id": tryon_id, "status": "processing completed"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# async def get_tryon_result(tryon_id: str):
#     result = await tryon_results_collection.find_one({"_id": tryon_id})
#     if not result:
#         raise HTTPException(status_code=404, detail="Result not found")
#     return result


# model = model_utils.load_dior_model()


# class ClothController:
#     def __init__(self):
#         self.model = model_utils.load_dior_model()
        
#     async def generate_outfit(
#         self, 
#         person_img: UploadFile, 
#         cloth_img: UploadFile,
#         pose_data: dict = None
#     ):
#         try:
#             # Process input images
#             person = await image_utils.process_image(person_img)
#             cloth = await image_utils.process_image(cloth_img)
            
#             # Run model inference
#             result = self._run_inference(person, cloth, pose_data)
            
#             # Post-process result
#             return image_utils.encode_image(result)
#         except Exception as e:
#             logger.error(f"Generation failed: {str(e)}")
#             raise HTTPException(500, "Outfit generation failed") from e

#     def _run_inference(
#         self, 
#         person: np.ndarray, 
#         cloth: np.ndarray, 
#         pose_data: dict = None
#     ):
#         # Prepare model input
#         model_input = {
#             "person": person,
#             "cloth": cloth,
#             "pose": pose_data
#         }
        
#         # Execute model pipeline
#         return self.model.generate(model_input)

# # Singleton instance for reuse
# cloth_controller = ClothController()






