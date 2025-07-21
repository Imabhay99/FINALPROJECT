
# routes/dependencies.py
from fastapi import Depends, HTTPException, status, UploadFile
from fastapi.security import HTTPBearer
from config.settings import settings

security = HTTPBearer()

async def validate_image(file: UploadFile) -> UploadFile:
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image files are allowed"
        )
    return file

async def validate_api_key(credentials: HTTPBearer = Depends(security)) -> None:
    api_key = credentials.credentials
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )







# from fastapi import Depends, HTTPException, status
# from fastapi.security import HTTPBearer
# from ..utils.image_utils import validate_image_file

# # Security scheme
# security = HTTPBearer()

# async def validate_image(file: UploadFile) -> UploadFile:
#     """Validate uploaded image file"""
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
#             detail="Only image files are allowed"
#         )
#     return file

# async def validate_api_key(
#     credentials: HTTPBearer = Depends(security)
# ) -> None:
#     """Validate API key from header"""
#     # Implement your API key validation logic here
#     api_key = credentials.credentials
#     if api_key != "your-secret-key":
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid API Key"
#         )