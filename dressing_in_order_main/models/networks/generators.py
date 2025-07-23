import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import functools
from models.networks.base_networks import get_norm_layer

from .base_networks import *
from models.base_model import BaseModel
from models.networks.base_networks import ResnetGenerator as OriginalResnetGenerator

# Optional: patch global reference (used for monkey-patching if necessary)
import dressing_in_order_main.models.networks.generators as generators_module


class PatchedResnetGenerator(OriginalResnetGenerator):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None,
                 use_dropout=False, n_blocks=6, padding_type='reflect',
                 **kwargs):
        # Accept and store additional arguments safely (optional)
        self.norm_type = kwargs.get('norm_type', 'instance')
        self.relu_type = kwargs.get('relu_type', 'relu')
        self.use_attention = kwargs.get('use_attention', False)

        # ✅ Default normalization if not provided
        if norm_layer is None:
            norm_layer = get_norm_layer(norm_type=self.norm_type)

        # ✅ Call parent constructor with only valid args
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            padding_type=padding_type
        )
generators_module.ResnetGenerator = PatchedResnetGenerator



class Resnet9blocksGenerator(nn.Module):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, output_nc=3, use_dropout=False, n_blocks=9, opt=None, **kwargs):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Remove potential duplication
        kwargs.pop('n_blocks', None)

        self.netG = PatchedResnetGenerator(
            input_nc=img_nc,
            output_nc=output_nc,
            ngf=ngf,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            **kwargs
        )

    def set_input(self, input_data):
        self.real_A = input_data['A'].to(self.device)
        self.image_paths = input_data.get('A_paths', None)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        with torch.no_grad():
            self.forward()

# ✅ DIOR base generator
class BaseGenerator(nn.Module):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64,
                 n_human_parts=8, n_downsampling=2, n_style_blocks=4,
                 norm_type='instance', relu_type='relu'):
        super().__init__()
        self.n_style_blocks = n_style_blocks
        self.n_human_parts = n_human_parts

        self.to_emb = ContentEncoder(n_downsample=n_downsampling, input_dim=18, dim=ngf,
                                     norm='instance', activ=relu_type, pad_type='zero')
        self.to_rgb = Decoder(n_upsample=n_downsampling, n_res=6, dim=latent_nc, output_dim=3)
        self.style_blocks = nn.Sequential(*[
            StyleBlock(latent_nc, relu_type=relu_type) for _ in range(n_style_blocks)
        ])

        self.fusion = nn.Sequential(
            Conv2dBlock(latent_nc + 1, latent_nc, 3, 1, 1, norm_type, relu_type),
            Conv2dBlock(latent_nc, latent_nc * 4, 3, 1, 1, 'none', 'none'),
        )


# ✅ DIORGenerator
class DIORGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def attend_person(self, psegs, gmask):
        styles = [a for a, b in psegs]
        mask = [b for a, b in psegs]
        style_skin = sum(styles).view(styles[0].size(0), styles[0].size(1), -1).sum(-1)
        human_mask = sum(mask).float().detach()
        area = human_mask.view(human_mask.size(0), human_mask.size(1), -1).sum(-1) + 1e-5
        style_skin = (style_skin / area)[:, :, None, None]

        full_human_mask = sum([m.float() for m in mask[1:] + gmask]).detach()
        full_human_mask = (full_human_mask > 0).float()
        style_human = style_skin * full_human_mask
        style_human = self.fusion(torch.cat([style_human, full_human_mask], 1))
        style_bg = self.fusion(torch.cat([styles[0], mask[0]], 1))
        style = style_human * full_human_mask + (1 - full_human_mask) * style_bg
        return style

    def attend_garment(self, gsegs, alpha=0.5):
        ret = []
        styles = [a for a, b in gsegs]
        attns = [b for a, b in gsegs]

        for s, attn in zip(styles, attns):
            attn = (attn > alpha).float().detach()
            s = F.interpolate(s, (attn.size(2), attn.size(3)))
            mean_s = s.view(s.size(0), s.size(1), -1).mean(-1).unsqueeze(-1).unsqueeze(-1)
            s = s + mean_s
            s = self.fusion(torch.cat([s, attn], 1)) * attn
            ret.append(s)

        return ret, attns

    def forward(self, pose, psegs, gsegs, alpha=0.5):
        style_fabrics, g_attns = self.attend_garment(gsegs, alpha=alpha)
        style_human = self.attend_person(psegs, g_attns)
        pose = self.to_emb(pose)
        out = pose
        for k in range(self.n_style_blocks // 2):
            out = self.style_blocks[k](out, style_human)
        for i, attn in enumerate(g_attns):
            curr_mask = (attn > alpha).float().detach()
            exists = torch.sum(curr_mask.view(curr_mask.size(0), -1), 1) > 0
            exists = exists[:, None, None, None].float()
            attn = exists * curr_mask * attn
            for k in range(self.n_style_blocks // 2, self.n_style_blocks):
                base = out
                out = self.style_blocks[k](out, style_fabrics[i], cut=True)
                out = out * attn + base * (1 - attn)
        return self.to_rgb(out)


# ✅ DIORv1Generator (minor difference from DIORGenerator)
class DIORv1Generator(DIORGenerator):
    def attend_person(self, psegs, gmask):
        styles = [a for a, b in psegs]
        mask = [b for a, b in psegs]
        style_skin = sum(styles).view(styles[0].size(0), styles[0].size(1), -1).sum(-1)
        human_mask = sum(mask).float().detach()
        area = human_mask.view(human_mask.size(0), human_mask.size(1), -1).sum(-1) + 1e-5
        style_skin = (style_skin / area)[:, :, None, None]
        full_human_mask = sum([m.float() for m in mask[1:] + gmask]).detach()
        full_human_mask = (full_human_mask > 0).float()
        style_human = style_skin * full_human_mask + styles[0]  # difference here
        style_human = self.fusion(torch.cat([style_human, full_human_mask], 1))
        style_bg = self.fusion(torch.cat([styles[0], mask[0]], 1))
        return style_human * full_human_mask + (1 - full_human_mask) * style_bg
