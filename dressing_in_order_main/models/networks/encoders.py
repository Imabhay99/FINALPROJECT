import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility for building conv block
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type='batch', relu_type='relu'):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    
    if norm_type == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm_type == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu_type == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif relu_type == 'lrelu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    return nn.Sequential(*layers)

class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsample=2,
                 norm_type='instance', relu_type='relu',
                 init_type='normal', init_gain=0.02,
                 gpu_ids=[], use_dropout=False,
                 norm_layer=None):  # ✅ Accept extra arg
        super(ResnetEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []

        # Initial layer
        model += [conv_block(input_nc, ngf, kernel_size=7, stride=1, padding=3,
                             norm_type=norm_type, relu_type=relu_type)]

        # Downsampling layers
        in_features = ngf
        for _ in range(n_downsample):
            out_features = in_features * 2
            model += [conv_block(in_features, out_features, kernel_size=4, stride=2, padding=1,
                                 norm_type=norm_type, relu_type=relu_type)]
            in_features = out_features

        # Final conv to output
        model += [nn.Conv2d(in_features, output_nc, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
