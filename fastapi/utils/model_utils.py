

# utils/model_utils.py
import torch
from dressing-in-order-main.models import create_model
from dressing-in-order-main.options.test_options import TestOptions

_dior_model = None

def load_dior_model():
    global _dior_model
    if _dior_model is None:
        opt = TestOptions().parse()
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
        opt.isTrain = False

        _dior_model = create_model(opt)
        _dior_model.load_networks("latest")
        _dior_model.eval()

        if torch.cuda.is_available():
            dummy_input = torch.randn(1, 3, 512, 384).cuda()
            with torch.no_grad():
                _ = _dior_model(dummy_input)

    return _dior_model