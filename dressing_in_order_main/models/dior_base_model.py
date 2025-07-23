import torch
from models.base_model import BaseModel
from models import networks
import torch.nn.functional as F
from utils.util import tensor2im
import imageio
import os

class DIORBaseModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.n_human_parts = opt.n_human_parts
        self.n_style_blocks = opt.n_style_blocks
        self.isTrain = getattr(opt, 'isTrain', True)

        # Optional configs (should be before model init)
        opt.norm_type = getattr(opt, 'norm_type', 'batch')
        opt.relu_type = getattr(opt, 'relu_type', 'relu')
        opt.use_dropout = getattr(opt, 'use_dropout', False)
        opt.padding_type = getattr(opt, 'padding_type', 'reflect')

        if self.isTrain:
            self._init_loss(opt)

        self._init_models(opt)

        # netG (can be inside _init_models, but okay here too)
        self.netG = networks.define_G(
            img_nc=3,
            kpt_nc=opt.n_kpts,
            ngf=opt.ngf,
            latent_nc=opt.ngf * (2 ** 2),
            style_nc=opt.style_nc,
            n_style_blocks=opt.n_style_blocks,
            n_human_parts=opt.n_human_parts,
            netG=opt.netG,
            norm=opt.norm_type,
            relu_type=opt.relu_type,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            gpu_ids=self.gpu_ids,
            use_dropout=opt.use_dropout,
            padding_type=opt.padding_type
        )


    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--loss_coe_rec', type=float, default=2)
        parser.add_argument('--loss_coe_per', type=float, default=0.2)
        parser.add_argument('--loss_coe_sty', type=float, default=200)
        parser.add_argument('--loss_coe_GAN', type=float, default=1)
        parser.add_argument('--g2d_ratio', type=float, default=0.1)
        parser.add_argument('--segm_dataset', type=str, default="")
        parser.add_argument('--netE', type=str, default='adgan')
        parser.add_argument('--n_human_parts', type=int, default=8)
        parser.add_argument('--n_kpts', type=int, default=18)
        parser.add_argument('--n_style_blocks', type=int, default=4)
        parser.add_argument('--style_nc', type=int, default=64)
        return parser

    def _init_loss(self, opt):
        from models import external_functions
        self.loss_names = ["G_GAN_pose", "G_GAN_content", 
                           "D_real_pose", "D_fake_pose", "D_real_content", "D_fake_content",
                           "rec", "per", "sty"]

        if self.isTrain:
            self.log_loss_update(reset=True)
            self.loss_coe = {'rec': opt.loss_coe_rec, 
                             'per': opt.loss_coe_per, 
                             'sty': opt.loss_coe_sty,
                             'GAN': opt.loss_coe_GAN}

            self.criterionGAN = external_functions.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean').to(self.device)
            self.criterionMSE = torch.nn.MSELoss(reduction="mean").to(self.device)

    def _init_models(self, opt):
        self.model_names = ["E_attr", "G", "VGG"]
        self.frozen_models = ["VGG"]
        self.visual_names = ['from_img', 'fake_B', 'to_img']
        self.netVGG = networks.define_tool_networks(tool='vgg', load_ckpt_path="", gpu_ids=opt.gpu_ids)

        self.netG = networks.define_G(img_nc=3, kpt_nc=opt.n_kpts, ngf=opt.ngf, latent_nc=opt.ngf * 4,
                                      style_nc=opt.style_nc, n_style_blocks=opt.n_style_blocks, netG=opt.netG,
                                      norm=opt.norm_type, relu_type=opt.relu_type,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.netE_attr = networks.define_E(input_nc=3, output_nc=opt.style_nc, netE=opt.netE, ngf=opt.ngf,
                                           n_downsample=2, norm_type=opt.norm_type, relu_type=opt.relu_type,
                                           init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.model_names += ["D_pose", "D_content"]
            self.netD_pose = networks.define_D(opt.n_kpts + 3, opt.ndf, opt.netD, opt.n_layers_D,
                                               norm=opt.norm_type, use_dropout=not opt.no_dropout,
                                               init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
            self.netD_content = networks.define_D(3 + self.n_human_parts, opt.ndf, opt.netD, n_layers_D=3,
                                                  norm=opt.norm_type, use_dropout=not opt.no_dropout,
                                                  init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

    def set_input(self, input):
        from_img, from_kpt, from_parse, to_img, to_kpt, to_parse, attr_label = input
        self.to_parse = to_parse.float().to(self.device)
        self.from_img = from_img.to(self.device)
        self.to_img = to_img.to(self.device)
        self.from_parse = from_parse.to(self.device)
        self.to_kpt = to_kpt.float().to(self.device)
        self.from_kpt = from_kpt.float().to(self.device)
        self.attr_label = attr_label.long().to(self.device)
        self.to_parse2 = torch.cat([(self.to_parse == i).unsqueeze(1) for i in range(self.n_human_parts)], 1).float()

    def encode_single_attr(self, img, parse, from_pose=None, to_pose=None, i=0):
        self.netE_attr.eval()
        with torch.no_grad():
            style_code = self.netE_attr(img)
        return style_code

    def encode_attr(self, img, parse, from_pose=None, to_pose=None):
        self.netE_attr.eval()
        with torch.no_grad():
            style_code = self.netE_attr(img)
        return style_code

    def decode(self, pose, attr_maps, attr_codes):
        self.netG.eval()
        with torch.no_grad():
            fake_img = self.netG(pose, attr_maps, attr_codes)
        return fake_img

    def forward(self):
        attr_code = self.encode_attr(self.from_img, self.from_parse)
        self.fake_B = self.decode(self.to_kpt, self.to_parse2, attr_code)

    def backward_D(self):
        self.loss_D = self.compute_D_pose_loss() + self.compute_D_content_loss()

    def compute_D_pose_loss(self):
        fake_AB = torch.cat((self.to_kpt, self.fake_B), 1)
        pred_fake = self.netD_pose(fake_AB.detach())
        self.loss_D_fake_pose = self.criterionGAN(pred_fake, False) * self.loss_coe['GAN']

        real_AB = torch.cat((self.to_kpt, self.to_img), 1)
        pred_real = self.netD_pose(real_AB)
        self.loss_D_real_pose = self.criterionGAN(pred_real, True) * self.loss_coe['GAN']

        return (self.loss_D_fake_pose + self.loss_D_real_pose) / 0.5

    def compute_D_content_loss(self):
        fake_AB = torch.cat((self.to_parse2, self.fake_B), 1)
        pred_fake = self.netD_content(fake_AB.detach())
        self.loss_D_fake_content = self.criterionGAN(pred_fake, False) * self.loss_coe['GAN']

        real_AB = torch.cat((self.to_parse2, self.to_img), 1)
        pred_real = self.netD_content(real_AB)
        self.loss_D_real_content = self.criterionGAN(pred_real, True) * self.loss_coe['GAN']

        return (self.loss_D_fake_content + self.loss_D_real_content) / 0.5

    def backward_G(self):
        fake_AB = torch.cat((self.to_kpt, self.fake_B), 1)
        pred_fake = self.netD_pose(fake_AB)
        self.loss_G_GAN_pose = self.criterionGAN(pred_fake, True)

        fake_AB = torch.cat((self.to_parse2, self.fake_B), 1)
        pred_fake = self.netD_content(fake_AB)
        self.loss_G_GAN_content = self.criterionGAN(pred_fake, True)

        self.loss_G = (self.loss_G_GAN_pose + self.loss_G_GAN_content) * self.loss_coe['GAN']

    def compute_rec_loss(self, pred, gt):
        self.loss_rec = 0.0
        if self.loss_coe['rec']:
            self.loss_rec = self.criterionL1(pred, gt) * self.loss_coe['rec']
        return self.loss_rec

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD_pose, True)
        self.set_requires_grad(self.netD_content, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD_pose, False)
        self.set_requires_grad(self.netD_content, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.log_loss_update()

    def save_batch(self, save_dir, count):
        rets = torch.cat([self.from_img, self.to_img, self.fake_B], 3)
        for i, ret in enumerate(rets):
            count += 1
            img = tensor2im(ret)
            imageio.imwrite(os.path.join(save_dir, f"generated_{count}.jpg"), img)
        return count
