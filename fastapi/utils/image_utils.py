import os
import uuid
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