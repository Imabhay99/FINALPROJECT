import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import random
from models.networks.base_networks import *
import os 
from models.base_model import BaseModel
from models.networks.base_networks import ResnetGenerator
import torch
from dressing_in_order_main.models.networks import generators

# Monkey-patch the ResnetGenerator to match the expected parameters
OriginalResnetGenerator = generators.ResnetGenerator

class Resnet9blocksGenerator(nn.Module):
    def __init__(self, img_nc, kpt_nc, style_nc, ngf=64, norm_type='batch', 
                 use_dropout=False, n_blocks=9, padding_type='reflect', 
                 relu_type='relu', use_attn=False, latent_nc=256, opt=None):
        super(Resnet9blocksGenerator, self).__init__()
        self.img_nc = img_nc
        self.kpt_nc = kpt_nc
        self.style_nc = style_nc
        self.latent_nc = latent_nc
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResnetGenerator(
            input_nc=img_nc,          # typically 3 for RGB
            output_nc=img_nc,         # usually same as input channels
            ngf=ngf,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            padding_type=padding_type,
            relu_type=relu_type
        ).to(self.device)

    def set_input(self, input_data):
        self.real_A = input_data['A'].to(self.device)  # Person image
        self.real_B = input_data.get('B', None)  # Optional: ground truth clothing
        self.image_paths = input_data.get('A_paths', None)

    def forward(self):
        self.fake_B = self.model(self.real_A)

    def optimize_parameters(self):
        # In inference, we just do forward pass
        with torch.no_grad():
            self.forward()

    def __call__(self, input_tensor):
        self.set_input({'A': input_tensor})
        self.optimize_parameters()
        return self.fake_B
    

class BaseGenerator(nn.Module):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64, n_human_parts=8, n_downsampling=2, n_style_blocks=4, norm_type='instance', relu_type='relu'):
        super(BaseGenerator, self).__init__()
        self.n_style_blocks = n_style_blocks
        self.n_human_parts = n_human_parts

        self.to_emb = ContentEncoder(n_downsample=n_downsampling, input_dim=18, dim=ngf,norm='instance', activ=relu_type, pad_type='zero')
        self.to_rgb = Decoder(n_upsample=n_downsampling, n_res=6, dim=latent_nc, output_dim=3)
        self.style_blocks = nn.Sequential(*[StyleBlock(latent_nc, relu_type=relu_type) for i in range(n_style_blocks)])

        self.fusion = nn.Sequential(
            Conv2dBlock(latent_nc + 1, latent_nc, 3, 1, 1, norm_type, relu_type),
            Conv2dBlock(latent_nc, latent_nc * 4, 3, 1, 1, 'none', 'none'),
            )

class DIORGenerator(BaseGenerator):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64, n_human_parts=4, n_downsampling=2, n_style_blocks=2, norm_type='instance', relu_type='relu', **kwargs):
        super(DIORGenerator, self).__init__(img_nc, kpt_nc, ngf, latent_nc, style_nc, 
        n_human_parts, n_downsampling, n_style_blocks, norm_type, relu_type, **kwargs)
      
        

    def attend_person(self, psegs, gmask):
        styles = [a for a,b in psegs]
        mask = [b for a,b in psegs]
        
        N,C,_,_ = styles[0].size()
        style_skin = sum(styles).view(N,C,-1).sum(-1)
        
        N,C,_,_ = mask[0].size()
        human_mask = sum(mask).float().detach()
        area = human_mask.view(N,C,-1).sum(-1) + 1e-5
        style_skin = (style_skin / area)[:,:,None,None]
        
        full_human_mask = sum([m.float() for m in mask[1:] + gmask]).detach() 
        full_human_mask = (full_human_mask > 0).float()
        style_human =  style_skin * full_human_mask # + styles[0]
        style_human = self.fusion(torch.cat([style_human, full_human_mask], 1))
        style_bg = self.fusion(torch.cat([styles[0], mask[0]], 1))        
        style = style_human * full_human_mask + (1 - full_human_mask) * style_bg 
        
        return style


    def attend_garment(self, gsegs, alpha=0.5):

        ret = []
        styles = [a for a,b in gsegs]
        attns = [b for a,b in gsegs]
        
        for s,attn in zip(styles, attns):
            attn = (attn > alpha).float().detach()
            s = F.interpolate(s, (attn.size(2), attn.size(3)))
            N,C,H,W = s.size()
            mean_s = s.view(N,C,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
            s = s + mean_s
            s = self.fusion(torch.cat([s, attn], 1))
            s = s * attn
            ret.append(s)
            
        return ret, attns

    def forward(self, pose, psegs, gsegs, alpha=0.5):

        style_fabrics, g_attns = self.attend_garment(gsegs, alpha=alpha)
        style_human  = self.attend_person(psegs, g_attns)
        
        pose = self.to_emb(pose)
        out = pose
        for k in range(self.n_style_blocks // 2):
            out = self.style_blocks[k](out, style_human)

        base = out
        self.base = base

        for i in range(len(g_attns)): 
            attn = g_attns[i]
            curr_mask = (attn > alpha).float().detach()
            N = curr_mask.size(0)
            exists = torch.sum(curr_mask.view(N,-1), 1)
            exists = (exists > 0)[:,None,None,None].float()
            
            attn = exists * curr_mask * attn # * fattn
            
            for k in range(self.n_style_blocks // 2, self.n_style_blocks):
                base0 = out
                out = self.style_blocks[k](out, style_fabrics[i],cut=True) 
                out = out * attn + base0 * (1 - attn)
                
            
        fake = self.to_rgb(out)
        return fake
    
class DIORv1Generator(BaseGenerator):
    def __init__(self, img_nc=3, kpt_nc=18, ngf=64, latent_nc=256, style_nc=64, n_human_parts=4, n_downsampling=2, n_style_blocks=2, norm_type='instance', relu_type='relu', **kwargs):
        super(DIORv1Generator, self).__init__(img_nc, kpt_nc, ngf, latent_nc, style_nc, 
        n_human_parts, n_downsampling, n_style_blocks, norm_type, relu_type, **kwargs)
      
        

    def attend_person(self, psegs, gmask):
        styles = [a for a,b in psegs]
        mask = [b for a,b in psegs]
        
        N,C,_,_ = styles[0].size()
        style_skin = sum(styles).view(N,C,-1).sum(-1)
        
        N,C,_,_ = mask[0].size()
        human_mask = sum(mask).float().detach()
        area = human_mask.view(N,C,-1).sum(-1) + 1e-5
        style_skin = (style_skin / area)[:,:,None,None]
        
        full_human_mask = sum([m.float() for m in mask[1:] + gmask]).detach() 
        full_human_mask = (full_human_mask > 0).float()
        style_human =  style_skin * full_human_mask + styles[0] # typo here
        style_human = self.fusion(torch.cat([style_human, full_human_mask], 1))
        style_bg = self.fusion(torch.cat([styles[0], mask[0]], 1))        
        style = style_human * full_human_mask + (1 - full_human_mask) * style_bg 
        
        return style


    def attend_garment(self, gsegs, alpha=0.5):

        ret = []
        styles = [a for a,b in gsegs]
        attns = [b for a,b in gsegs]
        
        for s,attn in zip(styles, attns):
            attn = (attn > alpha).float().detach()
            s = F.interpolate(s, (attn.size(2), attn.size(3)))
            N,C,H,W = s.size()
            mean_s = s.view(N,C,-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
            s = s + mean_s
            s = self.fusion(torch.cat([s, attn], 1))
            s = s * attn
            ret.append(s)
            
        return ret, attns

    def forward(self, pose, psegs, gsegs, alpha=0.5):

        style_fabrics, g_attns = self.attend_garment(gsegs, alpha=alpha)
        style_human  = self.attend_person(psegs, g_attns)
        
        pose = self.to_emb(pose)
        out = pose
        for k in range(self.n_style_blocks // 2):
            out = self.style_blocks[k](out, style_human)

        base = out
        self.base = base

        for i in range(len(g_attns)): 
            attn = g_attns[i]
            curr_mask = (attn > alpha).float().detach()
            N = curr_mask.size(0)
            exists = torch.sum(curr_mask.view(N,-1), 1)
            exists = (exists > 0)[:,None,None,None].float()
            
            attn = exists * curr_mask * attn # * fattn
            
            for k in range(self.n_style_blocks // 2, self.n_style_blocks):
                base0 = out
                out = self.style_blocks[k](out, style_fabrics[i],cut=True) 
                out = out * attn + base0 * (1 - attn)
                
            
        fake = self.to_rgb(out)
        return fake
    

class PatchedResnetGenerator(OriginalResnetGenerator):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2, **kwargs):
        # Filter out unexpected arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['norm_type', 'relu_type', 'use_attention']}
        
        # Call original constructor with only expected parameters
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            padding_type=padding_type,
            n_downsampling=n_downsampling
        )
generators.ResnetGenerator = PatchedResnetGenerator