# Virtual Clothing Try-On Setup Script
Write-Host "Starting setup for Virtual Clothing Try-On API..."

# 1. Delete existing virtual environment if present
if (Test-Path venv) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force venv
}

# 2. Create new virtual environment
Write-Host "Creating new virtual environment..."
python -m venv venv

# 3. Activate environment
Write-Host "Activating environment..."
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip to stable version
Write-Host "Upgrading pip..."
python -m pip install --upgrade "pip==23.2.1"

# 5. Create requirements files
Write-Host "Creating requirements files..."

# Create requirements_core.txt
@"
python-dotenv==1.0.0
pyyaml==6.0.1
wheel==0.42.0
setuptools==68.2.2
"@ | Out-File requirements_core.txt -Encoding utf8

# Create requirements_model.txt
@"
torch==2.1.2
torchvision==0.16.2
numpy==1.26.3
opencv-python-headless==4.9.0.80
scikit-image==0.22.0
scipy==1.11.4
tqdm==4.66.1
dominate==2.8.0
visdom==0.2.4
wandb==0.16.2
gdown==5.1.0
pillow==10.2.0
matplotlib==3.8.2
"@ | Out-File requirements_model.txt -Encoding utf8

# Create requirements_api.txt
@"
fastapi==0.109.1
uvicorn[standard]==0.27.0
motor==3.3.2
pydantic==2.6.1
pydantic-settings==2.1.0
python-multipart==0.0.6
cloudinary==1.37.0
requests==2.31.0
aiofiles==23.2.1
python-magic-bin==0.4.14
"@ | Out-File requirements_api.txt -Encoding utf8

# 6. Install PyTorch CPU separately
Write-Host "Installing PyTorch CPU..."
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# 7. Install other dependencies
Write-Host "Installing core dependencies..."
python -m pip install -r requirements_core.txt

Write-Host "Installing model dependencies..."
python -m pip install -r requirements_model.txt

Write-Host "Installing API dependencies..."
python -m pip install -r requirements_api.txt

# 8. Create necessary directories
Write-Host "Creating project directories..."
New-Item -ItemType Directory -Path static -Force
New-Item -ItemType Directory -Path static\temp -Force
New-Item -ItemType Directory -Path static\results -Force
New-Item -ItemType Directory -Path config -Force
New-Item -ItemType Directory -Path controllers -Force
New-Item -ItemType Directory -Path routes -Force
New-Item -ItemType Directory -Path utils -Force

# 9. Create .env file with your credentials
Write-Host "Creating .env file..."
@"
MONGO_URI=mongodb+srv://abhaymishra9945:AeK5ji2BehQ8O5kb@cluster0.wmoc0pi.mongodb.net/cloth_visually_try?retryWrites=true&w=majority&appName=Cluster0
CLOUDINARY_CLOUD_NAME=drxs4yu5j
CLOUDINARY_API_KEY=792394351774396
CLOUDINARY_API_SECRET=FlYEHsvl6g_sKNSm8P46hf1GdYg
JWT_SECRET=cloth_visually_try
MODEL_CHECKPOINT_PATH=./dressing-in-order-main/checkpoints/latest_net_G.pth
"@ | Out-File .env -Encoding utf8

# 10. Verify installation
Write-Host "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from fastapi import FastAPI; print('FastAPI installation verified')"

Write-Host "`nSetup completed successfully!`n"
Write-Host "To activate the virtual environment, run:"
Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "`nTo start the API server, run:"
Write-Host "    python main.py" -ForegroundColor Green