import torch
import torch.nn as nn

class LocalAttnReshape(nn.Module):
    def __init__(self):
        super(LocalAttnReshape, self).__init__()

    def forward(self, inputs, kernel_size=3):
        """
        inputs: shape (B, C, H, W) where C = kernel_size * kernel_size
        output: shape (B, 1, kernel_size*H, kernel_size*W)
        """
        B, C, H, W = inputs.size()
        assert C == kernel_size * kernel_size, f"Expected channel size {kernel_size*kernel_size}, got {C}"

        # Create an empty output tensor
        out_H, out_W = kernel_size * H, kernel_size * W
        output = torch.zeros((B, 1, out_H, out_W), dtype=inputs.dtype, device=inputs.device)

        for b in range(B):
            for c in range(C):
                row_offset = c // kernel_size
                col_offset = c % kernel_size
                output[b, 0, row_offset::kernel_size, col_offset::kernel_size] = inputs[b, c, :, :]

        return output

