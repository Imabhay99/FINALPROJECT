from resample2d import Resample2d
import torch
import numpy as np
import imageio

def tensor2im(tensor):
    img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img + 1) / 2 * 255.0
    return img.clip(0, 255).astype(np.uint8)

def save_image(img_np, path):
    imageio.imwrite(path, img_np)

if __name__ == "__main__":
    model = Resample2d()
    B, C, H, W = 1, 3, 64, 64

    input1 = torch.randn(B, C, H, W)
    flow = torch.zeros(B, 2, H, W)
    flow[:, 0, :, :] += 5.0  # shift right by 5 pixels
    flow[:, 1, :, :] += 3.0  # shift down by 3 pixels

    output = model(input1, flow)
    print("Output shape:", output.shape)

    save_image(tensor2im(input1), "original.png")
    save_image(tensor2im(output), "resampled.png")
