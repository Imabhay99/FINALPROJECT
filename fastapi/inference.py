import sys
import os
import torch
import numpy as np
import cv2
from config.settings import settings

# === Set PYTHONPATH correctly ===
sys.path.insert(0, r"D:\backends")

# === Define model directory and add to sys.path ===
model_dir = r"D:\backends\dressing_in_order_main"
if os.path.exists(model_dir):
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        print(f"📂 Added to sys.path: {model_dir}")
else:
    raise ImportError(f"❌ Model directory not found at: {model_dir}")

# === Import and patch generators BEFORE using DIORModel ===
try:
    from dressing_in_order_main.models.networks import generators
    print("✅ Successfully imported generators module.")
except ImportError as e:
    print(f"❌ Failed to import generators module: {e}")
    raise

print("🔧 Patching ResnetGenerator...")
OriginalResnetGenerator = generators.ResnetGenerator

class PatchedResnetGenerator(OriginalResnetGenerator):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None,
                 use_dropout=False, n_blocks=6, padding_type='reflect',
                 n_downsampling=2, relu_type='relu', norm_type='instance', **kwargs):

        self.relu_type = relu_type
        self.norm_type = norm_type

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

generators.ResnetGenerator = PatchedResnetGenerator
print("✅ ResnetGenerator patched.")

# === Now import DIORModel ===
try:
    from dressing_in_order_main.models.dior_model import DIORModel
    print("✅ Successfully imported DIORModel.")
except ImportError as e:
    print(f"❌ Failed to import DIORModel: {e}")
    raise

# === Define Mock Options ===
class MockOptions:
    def __init__(self):
        self.isTrain = False
        self.gpu_ids = []
        self.batch_size = 1
        self.checkpoints_dir = os.path.dirname(settings.MODEL_CHECKPOINT_PATH)
        self.name = os.path.basename(os.path.dirname(settings.MODEL_CHECKPOINT_PATH))
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
        self.n_style_blocks = 0
        self.norm_type = 'instance'
        self.relu_type = 'leakyrelu'
        self.use_dropout = False
        self.padding_type = 'reflect'
        self.n_downsampling = 2
        self.n_blocks = 9

    def __getattr__(self, name):
        if name.startswith('n_'):
            return 0
        elif name.startswith('use_'):
            return False
        elif name.endswith('_nc'):
            return 3
        elif name.endswith('_size'):
            return 256
        print(f"⚠️ Warning: Missing attribute '{name}', returning None.")
        return None

# === Load the DIOR model ===
def load_model():
    opt = MockOptions()

    # ✅ Ensure load_iter is not None
    if not hasattr(opt, 'load_iter') or opt.load_iter is None:
        opt.load_iter = 0

    model = DIORModel(opt)

    if hasattr(model, 'setup'):
        model.setup(opt)

    checkpoint = torch.load(settings.MODEL_CHECKPOINT_PATH, map_location=torch.device('cpu'))

    if hasattr(model, 'load_networks'):
        model.load_networks(opt.epoch_count)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
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


# === Preprocess image ===
def preprocess_image(image: np.ndarray, target_size=(192, 256)):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image).unsqueeze(0)

# === Run inference ===
def run_inference(model, cloth_img: np.ndarray, user_img: np.ndarray, pose_img: np.ndarray) -> np.ndarray:
    cloth_tensor = preprocess_image(cloth_img)
    user_tensor = preprocess_image(user_img)
    pose_tensor = preprocess_image(pose_img)

    model.eval()

    with torch.no_grad():
        if hasattr(model, 'inference'):
            output = model.inference(cloth_tensor, user_tensor, pose_tensor)
        elif hasattr(model, 'forward'):
            output = model.forward(cloth_tensor, user_tensor, pose_tensor)
        else:
            output = model(cloth_tensor, user_tensor, pose_tensor)

    if isinstance(output, tuple):
        output = output[0]

    output = output.squeeze().cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = (output + 1.0) * 127.5
    return np.clip(output, 0, 255).astype(np.uint8)
