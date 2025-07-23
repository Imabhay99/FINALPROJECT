from block_extractor import BlockExtractor
import torch
from PIL import Image
import torchvision.transforms as transforms
import imageio
import numpy as np

def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    image_numpy = image_tensor[0, :3].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    imageio.imwrite(image_path, image_numpy)

if __name__ == "__main__":
    extractor = BlockExtractor(kernel_size=3)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open("test.jpg")  # Replace with actual image path
    source = transform(image).unsqueeze(0)  # (1, 3, H, W)
    flow = torch.zeros((1, 2, 16, 16))  # Small zero flow

    output = extractor(source, flow)
    save_image(tensor2im(output), "output.png")


