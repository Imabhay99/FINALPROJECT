
from fastapi import APIRouter, UploadFile, File
from controllers.cloth_controller import (
    upload_cloth_image,
    upload_user_photo,
    generate_tryon,
    get_tryon_result,
    list_clothes
)

router = APIRouter()

@router.get("/")
async def list_clothes_route(skip: int = 0, limit: int = 10):
    return await list_clothes(skip, limit)

@router.post("/upload-cloth")
async def upload_cloth_route(
    name: str,
    category: str,
    image: UploadFile = File(...),
    description: str = ""
):
    return await upload_cloth_image(name, category, description, image)

@router.post("/upload-user-photo")
async def upload_user_photo_route(
    user_image: UploadFile = File(...),
    pose_image: UploadFile = File(...)
):
    return await upload_user_photo(user_image, pose_image)

@router.post("/try-on/{cloth_id}")
async def try_on_route(cloth_id: str, user_photo_id: str):
    return await generate_tryon(cloth_id, user_photo_id)

@router.get("/result/{tryon_id}")
async def get_result_route(tryon_id: str):
    return await get_tryon_result(tryon_id)








# from flask import Blueprint, request, jsonify
# from controllers import process_try_on
# import cloudinary.uploader

# app = Blueprint('routes', __name__)

# @app.route('/api/try-on', methods=['POST'])
# def try_on():
#     if 'user_photo' not in request.files or 'cloth_photo' not in request.files:
#         return jsonify({'error': 'Both user and cloth photos are required'}), 400

#     user_photo = request.files['user_photo']
#     cloth_photo = request.files['cloth_photo']

#     # Upload to Cloudinary
#     user_upload = cloudinary.uploader.upload(user_photo, folder='try_on/users')
#     cloth_upload = cloudinary.uploader.upload(cloth_photo, folder='try_on/clothes')

#     user_url = user_upload['secure_url']
#     cloth_url = cloth_upload['secure_url']

#     # Process with the model
#     result_url = process_try_on(user_url, cloth_url)

#     return jsonify({'result_url': result_url}), 200








# # routes/cloth_routes.py
# from fastapi import APIRouter, UploadFile, File, Form
# from controllers.cloth_controller import process_virtual_tryon

# router = APIRouter()

# @router.post("/tryon")
# async def tryon_cloth(
#     prompt: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     return await process_virtual_tryon(image, prompt)

# # routes/cloth_routes.py
# from fastapi import APIRouter, UploadFile, File, Form
# from controllers.cloth_controller import process_virtual_tryon

# router = APIRouter()

# @router.post("/tryon")
# async def tryon_cloth(
#     prompt: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     return await process_virtual_tryon(image, prompt)

