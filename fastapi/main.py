from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes import cloth_routes
from config.settings import settings
import os

app = FastAPI(title="Virtual Clothing Try-On API")

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
app.include_router(cloth_routes.router, prefix="/api/v1/clothes", tags=["clothes"])

# Serve static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

