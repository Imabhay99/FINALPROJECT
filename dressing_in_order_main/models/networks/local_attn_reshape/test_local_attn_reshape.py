from local_attn_reshape import LocalAttnReshape
import torch
import numpy as np
from PIL import Image
import imageio

def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    image_numpy = image_tensor[0, :1].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    imageio.imwrite(image_path, image_numpy)

if __name__ == "__main__":
    extractor = LocalAttnReshape()
    kernel_size = 3
    B, H, W = 1, 5, 5
    C = kernel_size * kernel_size

    # Create test tensor with channels = kernel_size^2
    test_input = torch.arange(B * C * H * W).view(B, C, H, W).float()
    output = extractor(test_input, kernel_size=kernel_size)
    print("Output shape:", output.shape)  # Should be [B, 1, kernel_size*H, kernel_size*W]
    print("Sample output:\n", output[0, 0, :6, :6])
