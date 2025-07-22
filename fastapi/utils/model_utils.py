import os
import torch
import logging
from dressing_in_order_main.models import create_model
from dressing_in_order_main.options.test_options import TestOptions
from config.settings import settings

logger = logging.getLogger(__name__)

_dior_model = None

def load_dior_model():
    global _dior_model
    if _dior_model is not None:
        return _dior_model

    try:
        logger.info("Loading DIOR model...")
        
        # Parse default options
        opt = TestOptions().parse([])  # Pass empty list to avoid CLI arguments
        
        # Configure device settings
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            opt.gpu_ids = [0]
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            opt.gpu_ids = []
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        opt.isTrain = False
        
        # Configure model paths from settings
        checkpoint_dir = os.path.dirname(settings.MODEL_CHECKPOINT_PATH)
        opt.checkpoints_dir = checkpoint_dir
        opt.name = os.path.basename(checkpoint_dir)
        
        logger.debug(f"Model options: {opt}")
        
        # Create model
        _dior_model = create_model(opt)
        
        # Load checkpoint
        _dior_model.load_networks("latest")
        _dior_model.eval()
        _dior_model = _dior_model.to(device)
        
        # Warm-up run to initialize model
        logger.info("Running warm-up inference...")
        dummy_input = torch.randn(1, 3, 512, 384).to(device)
        with torch.no_grad():
            _ = _dior_model(dummy_input)
        
        logger.info("DIOR model loaded successfully")
        return _dior_model
        
    except Exception as e:
        logger.exception(f"Failed to load DIOR model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")