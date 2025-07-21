
# routes/cloth_routes.py
from fastapi import APIRouter, UploadFile, File, Depends
from controllers.cloth_controller import (
    upload_cloth_image,
    upload_user_photo,
    generate_tryon,
    get_tryon_result,
    list_clothes,
    cloth_controller
)
from routes.dependencies import validate_image

router = APIRouter()

@router.get("/")
async def list_clothes_route(skip: int = 0, limit: int = 10):
    return await list_clothes(skip, limit)

@router.post("/upload-cloth")
async def upload_cloth_route(name: str, category: str, image: UploadFile = File(...), description: str = ""):
    return await upload_cloth_image(name, category, description, image)

@router.post("/upload-user-photo")
async def upload_user_photo_route(user_image: UploadFile = File(...), pose_image: UploadFile = File(...)):
    return await upload_user_photo(user_image, pose_image)

@router.post("/try-on/{cloth_id}")
async def try_on_route(cloth_id: str, user_photo_id: str):
    return await generate_tryon(cloth_id, user_photo_id)

@router.get("/result/{tryon_id}")
async def get_result_route(tryon_id: str):
    return await get_tryon_result(tryon_id)

@router.post("/generate-outfit")
async def generate_outfit(
    person_img: UploadFile = File(..., dependencies=[Depends(validate_image)]),
    cloth_img: UploadFile = File(..., dependencies=[Depends(validate_image)])
):
    result = await cloth_controller.generate_outfit(person_img, cloth_img)
    return {"result": result}

@router.post("/generate-with-pose")
async def generate_with_pose(
    person_img: UploadFile = File(..., dependencies=[Depends(validate_image)]),
    cloth_img: UploadFile = File(..., dependencies=[Depends(validate_image)]),
    pose_data: dict = {}
):
    result = await cloth_controller.generate_outfit(person_img, cloth_img, pose_data)
    return {"result": result}

