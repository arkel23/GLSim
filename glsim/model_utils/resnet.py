import os
import sys
import errno
import math
import warnings

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import get_dir, download_url_to_file, _is_legacy_zip_format, _legacy_zip_load
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange, Reduce


URL = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'


def load_state_dict(model_name='resnet50_miil_21k', map_location='cpu'):
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    # cached_file doesnt have extension so it doesnt differentiate between pth and npz files
    cached_file = os.path.join(model_dir, model_name)
    cached_file_pth = '{}.pth'.format(cached_file)
    if not os.path.exists(cached_file_pth):
        sys.stderr.write(
            'Downloading: "{}" to {}\n'.format(URL, cached_file_pth))
        download_url_to_file(URL, cached_file_pth, progress=True)
    if _is_legacy_zip_format(cached_file_pth):
        return _legacy_zip_load(cached_file_pth, model_dir, map_location)

    fp = f'{os.path.join(model_dir, model_name)}.pth'
    if not os.path.exists(fp):
        raise FileNotFoundError

    return torch.load(fp, map_location=map_location)


class Resnet(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.debugging = config.debugging

        if 'resnet50' in config.model_name:
            self.encoder = timm.create_model('resnet50', pretrained=pretrained, num_classes=0, global_pool='')
            # https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/MODEL_ZOO.md
            if pretrained and 'in21k' in config.model_name:
                state_dict = load_state_dict()
                ret = self.encoder.load_state_dict(state_dict['state_dict'], strict=False)
                print('Loaded ImageNet-21k ckpt', ret)
        else:
            self.encoder = timm.create_model(config.model_name,
                                             pretrained=pretrained, num_classes=0, global_pool='')

        # Classifier head
        if config.load_fc_layer:
            if config.classifier == 'pool':
                self.head_pool = Reduce('b s d -> b d', 'mean')
                self.head = nn.Linear(config.representation_size, config.num_classes)

        # Initialize weights
        self.init_weights()

    def get_anchor_pool(self, anchor_size):
        if self.anchor_pool_avg:
            pool = nn.AvgPool2d(kernel_size=anchor_size // self.patch_equivalent,
                                stride=self.anchor_pool_stride, padding=0)
        else:
            pool = nn.MaxPool2d(kernel_size=anchor_size // self.patch_equivalent,
                                stride=self.anchor_pool_stride, padding=0)
        return pool

    @torch.no_grad()
    def init_weights(self):
        if hasattr(self, 'head'):
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x, ret_dist=False):
        """
        x (tensor): b k c fh fw -> b s d
        """
        x = self.encoder(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        if hasattr(self, 'head_pool'):
            x = self.head_pool(x)
            x = self.head(x)
        return x
