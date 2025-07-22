import sys
import os
import torch
import numpy as np
import cv2
from config.settings import settings

# --- FIRST: Add the model directory to Python path ---
# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the path to the model directory
model_dir = os.path.abspath(os.path.join(current_dir, 'dressing_in_order_main'))

# Add to Python path if it exists
if os.path.exists(model_dir):
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        print(f"Added to sys.path: {model_dir}")
else:
    raise ImportError(f"Model directory not found at {model_dir}")

# --- SECOND: Import ONLY the generators module for patching ---
try:
    # Try to import only the generators module
    from dressing_in_order_main.models.networks import generators
    print("Successfully imported generators")
except ImportError as e:
    print(f"Import failed: {str(e)}")
    # Try alternative import approach
    try:
        from dressing_in_order_main.models.networks import generators
        print("Successfully imported generators using direct path")
    except ImportError:
        print("All import attempts failed")
        raise

# --- THIRD: Apply the patch BEFORE importing DIORModel ---
print("Patching ResnetGenerator...")
OriginalResnetGenerator = generators.ResnetGenerator

class PatchedResnetGenerator(OriginalResnetGenerator):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2, **kwargs):
        # Filter out unexpected arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['norm_type', 'relu_type', 'use_attention']}
        
        # Call original constructor with only expected parameters
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            padding_type=padding_type,
            n_downsampling=n_downsampling
        )

# Apply the patch to the generators module
generators.ResnetGenerator = PatchedResnetGenerator
print("ResnetGenerator patched successfully")

# --- FOURTH: Now import DIORModel AFTER patching ---
try:
    # Now import DIORModel after patching
    from dressing_in_order_main.models.dior_model import DIORModel
    print("Successfully imported DIORModel")
except ImportError as e:
    print(f"DIORModel import failed: {str(e)}")
    # Try alternative import approach
    try:
        from dressing_in_order_main.models.dior_model import DIORModel
        print("Successfully imported DIORModel using direct path")
    except ImportError:
        print("DIORModel import failed")
        raise

# Create a comprehensive mock options object with dynamic attribute handling
class MockOptions:
    def __init__(self):
        # Set default values for known options
        self.isTrain = False
        self.gpu_ids = []
        self.batch_size = 1
        self.checkpoints_dir = os.path.dirname(settings.MODEL_CHECKPOINT_PATH)
        self.name = os.path.basename(os.path.dirname(settings.MODEL_CHECKPOINT_PATH))
        self.continue_train = False
        self.epoch_count = 1
        self.phase = 'test'
        self.serial_batches = True
        self.paired = True
        self.model = 'dior'
        self.norm = 'instance'
        self.no_dropout = True
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_flip = True
        self.direction = 'AtoB'
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.lambda_L1 = 100.0
        self.dataset_mode = 'keypoint'
        self.preprocess = 'resize'
        self.load_size = 256
        self.crop_size = 256
        self.max_dataset_size = float("inf")
        self.no_rotation = False
        self.n_human_parts = 8
        self.n_style = 8
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_local_enhancers = 1
        self.n_blocks_local = 3
        self.use_local = True
        self.use_style = True
        self.use_attention = True
        self.use_mask = True
        self.use_pose = True
        self.use_bg = False
        
        # Add the missing attribute from the error
        self.n_style_blocks = 0  # Default value
        
        # Add parameters that match the ResnetGenerator signature
        self.norm_type = 'instance'
        self.relu_type = 'leakyrelu'
        self.use_dropout = False
        self.padding_type = 'reflect'
        self.n_downsampling = 2
        self.n_blocks = 9
        
    def __getattr__(self, name):
        """Dynamically handle any missing attributes"""
        # Provide default values for common attribute patterns
        if name.startswith('n_'):
            return 0
        elif name.startswith('use_'):
            return False
        elif name.endswith('_nc'):
            return 3
        elif name.endswith('_size'):
            return 256
        
        # Warn about missing attribute but return None
        print(f"Warning: MockOptions missing attribute '{name}', returning None")
        return None

def load_model():
    """Load the DIOR model with CPU-compatible settings"""
    # Create mock options
    opt = MockOptions()
    
    # Initialize model
    model = DIORModel(opt)
    
    # Setup model (if required)
    if hasattr(model, 'setup'):
        model.setup(opt)
    
    # Load checkpoint
    checkpoint = torch.load(
        settings.MODEL_CHECKPOINT_PATH,
        map_location=torch.device('cpu')
    )
    
    # Handle different checkpoint formats
    if hasattr(model, 'load_networks'):
        # If the model has a built-in loading method
        model.load_networks(opt.epoch_count)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Try to load directly into the model
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            # Handle submodule loading
            if "Missing key(s)" in str(e):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                raise
    
    model.eval()
    return model

def preprocess_image(image: np.ndarray, target_size=(192, 256)):
    """Preprocess image for model input"""
    # Ensure image is in RGB format
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    # Normalize
    image = image.astype(np.float32) / 127.5 - 1.0
    # Convert to tensor (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image).unsqueeze(0)

def run_inference(model, cloth_img: np.ndarray, user_img: np.ndarray, pose_img: np.ndarray) -> np.ndarray:
    """Run inference using the DIOR model"""
    # Preprocess images
    cloth_tensor = preprocess_image(cloth_img)
    user_tensor = preprocess_image(user_img)
    pose_tensor = preprocess_image(pose_img)
    
    # Set to evaluation mode
    model.eval()
    
    # Run model
    with torch.no_grad():
        # Try different inference methods
        if hasattr(model, 'inference'):
            output = model.inference(cloth_tensor, user_tensor, pose_tensor)
        elif hasattr(model, 'forward'):
            output = model.forward(cloth_tensor, user_tensor, pose_tensor)
        else:
            output = model(cloth_tensor, user_tensor, pose_tensor)
    
    # Postprocess output
    if isinstance(output, tuple):
        output = output[0]  # Get the first output if multiple are returned
    
    output = output.squeeze().cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output