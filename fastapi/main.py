from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes import cloth_routes
from config.settings import settings
from routes.dependencies import validate_api_key  # Add this import
import os
import uvicorn

app = FastAPI(title="Fashion Generation API",
              description="Virtual try-on and outfit generation service",
              version="1.0.0",
              openapi_url="/api/openapi.json" if settings.DEBUG else None)

# Create directories
os.makedirs(settings.STATIC_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(
    cloth_routes.router, 
    prefix="/api/v1/clothes", 
    tags=["clothes"], 
    dependencies=[Depends(validate_api_key)]  # Now this will work
)

# Serve static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )