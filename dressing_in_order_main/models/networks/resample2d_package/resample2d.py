import torch
import torch.nn as nn
import torch.nn.functional as F

class Resample2d(nn.Module):
    def __init__(self, kernel_size=2, dilation=1, sigma=5):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = sigma

    def forward(self, input1, input2):
        """
        input1: (B, C, H, W)
        input2: (B, 2, H, W) - flow (dx, dy)
        Returns: warped version of input1
        """
        B, C, H, W = input1.size()

        # If sigma is needed: ignore for now
        if input2.size(1) == 3:
            input2 = input2[:, :2, :, :]  # Drop sigma (not used in CPU)

        dx = input2[:, 0, :, :]
        dy = input2[:, 1, :, :]

        # Create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1).to(input1.device)
        grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1).to(input1.device)

        # Apply flow
        xf = grid_x + dx
        yf = grid_y + dy

        # Normalize to [-1, 1] for grid_sample
        xf_norm = 2.0 * xf / (W - 1) - 1.0
        yf_norm = 2.0 * yf / (H - 1) - 1.0

        # Stack into grid shape
        flow_grid = torch.stack((xf_norm, yf_norm), dim=3)  # shape: (B, H, W, 2)

        # Sample
        output = F.grid_sample(input1, flow_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return output
