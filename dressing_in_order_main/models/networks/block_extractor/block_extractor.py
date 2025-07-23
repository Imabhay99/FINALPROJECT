import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockExtractor(nn.Module):  # same class name for compatibility
    def __init__(self, kernel_size=3):
        super(BlockExtractor, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, source, flow_field):
        B, C, H, W = source.size()
        _, _, Hf, Wf = flow_field.size()
        ks = self.kernel_size

        # Upsample flow to match source size
        flow_field = F.interpolate(flow_field, size=(H, W), mode='bilinear', align_corners=True)

        # Create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # shape: (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        grid = grid.to(source.device)  # ensure same device

        # Apply flow field to grid
        sampling_grid = grid + flow_field

        # Normalize to [-1, 1] for grid_sample
        sampling_grid[:, 0, :, :] = 2.0 * sampling_grid[:, 0, :, :] / (W - 1) - 1.0
        sampling_grid[:, 1, :, :] = 2.0 * sampling_grid[:, 1, :, :] / (H - 1) - 1.0
        sampling_grid = sampling_grid.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # Sample using grid_sample
        output = F.grid_sample(source, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return output
