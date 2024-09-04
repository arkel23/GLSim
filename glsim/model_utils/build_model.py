from types import SimpleNamespace
import timm
import torch
import torch.nn as nn

# from .pretrained_vit import *
from .glsim import ViTGLSim
from .timm_glsim import *
from .resnet import Resnet
from .ffvt import FFViT, CONFIGS_FFVT, load_state_dict
from .transfg import TFG, CONFIGS_TRANSFG
from .cal import CAL, CONFIGS_CAL
from .configs import ViTConfig, ResnetConfig


VITS = (
    'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32',
    'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_bs16', 'vit_b32',
    'vit_l16', 'vit_l32', 'vit_h14')


def build_model(args):
    # initiates model and loss
    if args.model_name in VITS:
        model = VisionTransformer(args)
    elif 'ffvt' in args.model_name:
        model = FFVT(args)
    elif 'transfg' in args.model_name:
        model = TransFG(args)
    elif 'cal' in args.model_name:
        model = CounterfactualAttentionLearning(args)
    elif 'resnet' in args.model_name:
        model = ResNet(args)
    elif args.model_name in timm.list_models(pretrained=True):
        model = TIMMViT(args)
    else:
        raise NotImplementedError

    args.seq_len_post_reducer = model.cfg.seq_len_post_reducer

    if args.ckpt_path:
        state_dict = torch.load(
            args.ckpt_path, map_location=torch.device('cpu'))['model']
        expected_missing_keys = []
        if args.transfer_learning:
            # modifications to load partial state dict
            if ('model.head.weight' in state_dict):
                expected_missing_keys += ['model.head.weight', 'model.head.bias']
            for key in expected_missing_keys:
                state_dict.pop(key)
        ret = model.load_state_dict(state_dict, strict=False)
        print('''Missing keys when loading pretrained weights: {}
              Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))
        print('Loaded from custom checkpoint.')

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.assertions_corrections()
        cfg.calc_dims()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViTGLSim(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, labels=None, train=False, ret_dist=False):
        if train or not self.cfg.test_flip:
            out = self.model(images, ret_dist=ret_dist)
        else:
            with torch.no_grad():
                out_o = self.model(images, ret_dist=ret_dist)
                images_m = torch.flip(images, [3])
                out_m = self.model(images_m, ret_dist=ret_dist)
                if isinstance(out_o, tuple) and len(out_o) == 4:
                    out = (out_o[0] + out_m[0]) / 2
                    out = [out, out_o[1], out_o[2], out_o[3]]
                elif isinstance(out_o, tuple) and len(out_o) == 3:
                    out = (out_o[0] + out_m[0]) / 2
                    out = [out, out_o[1], out_o[2]]
                elif isinstance(out_o, tuple) and len(out_o) == 2:
                    out = (out_o[0] + out_m[0]) / 2
                    out = [out, out_o[1]]
                else:
                    out = (out_o + out_m) / 2
        return out



class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        # init default config
        cfg = ResnetConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.adjust_resnet()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = Resnet(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, labels=None, train=False):
        out = self.model(images)
        return out


class FFVT(nn.Module):
    def __init__(self, args):
        super(FFVT, self).__init__()
        # init default config
        cfg = CONFIGS_FFVT['ffvt']
        cfg.sd = args.sd
        cfg.patchifier_dropout = args.patchifier_dropout

        if args.dataset_name == 'dogs':
            cfg.num_token = 24
        # cfg.num_token = 24 for dog, 12 for soy, cotton and cub
        # modify config if given an arg otherwise keep defaults

        self.model = FFViT(
            cfg, img_size=args.image_size, zero_head=True,
            num_classes=args.num_classes, vis=True)
        if args.pretrained:
            self.model.load_from(load_state_dict())
            print('Load pretrained model')
        self.cfg = cfg
        self.cfg.seq_len_post_reducer = (args.image_size // 16) ** 2

    def forward(self, images, labels=None, train=False):
        out = self.model(images)
        return out


class TransFG(nn.Module):
    def __init__(self, args):
        super(TransFG, self).__init__()
        # init default config
        cfg = CONFIGS_TRANSFG['transfg']
        cfg.sd = args.sd
        cfg.classifier_aux = 'cont'
        cfg.patchifier_dropout = args.patchifier_dropout

        self.model = TFG(
            cfg, img_size=args.image_size, zero_head=True,
            num_classes=args.num_classes)

        if args.pretrained:
            self.model.load_from(load_state_dict())
            print('Load pretrained model')

        self.cfg = cfg
        self.cfg.seq_len_post_reducer = (((args.image_size - 16) // cfg.slide_step) + 1) ** 2

    def forward(self, images, labels=None, train=False):
        out = self.model(images)
        return out


class CounterfactualAttentionLearning(nn.Module):
    def __init__(self, args):
        super(CounterfactualAttentionLearning, self).__init__()
        # init default config
        cfg = CONFIGS_CAL['cal']
        self.model = CAL(
            cfg, num_classes=args.num_classes, device=args.device,
            single_crop=args.cal_single_crop)
        # if args.pretrained:
        #    self.model.net.load_state_dict()
        #    print('Load pretrained model')
        self.cfg = cfg
        self.cfg.seq_len_post_reducer = (args.image_size // 32) ** 2

    def forward(self, images, labels=None, train=False, ret_dist=False):
        out = self.model(images, labels, train=train)
        return out


class TIMMViT(nn.Module):
    def __init__(self, args):
        super(TIMMViT, self).__init__()
        self.model = timm.create_model(args.model_name, num_classes=1000,
                                       img_size=args.image_size, drop_path_rate=args.sd,
                                       pretrained=args.pretrained, args=args)

        if args.classifier == 'cls_pool':
            global_pool = 'cls_pool'
        elif args.classifier == 'pool':
            global_pool = 'pool'
        else:
            global_pool = 'token'
        self.model.reset_classifier(args.num_classes, global_pool)

        patch_size = int([s for s in args.model_name.split('_') if 'patch' in s][0].replace('patch', ''))
        if args.dynamic_anchor:
            args.anchor_size = patch_size
            self.model.add_crop_cls()
        cfg = dict(seq_len_post_reducer=((args.image_size // patch_size) ** 2))
        self.cfg = SimpleNamespace(**cfg)

    def forward(self, images, labels=None, train=False):
        out = self.model(images)
        return out
