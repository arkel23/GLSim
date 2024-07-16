'''
https://github.com/raoyongming/CAL/tree/master/fgvc
https://github.com/raoyongming/CAL/blob/master/fgvc/train_distributed.py
https://github.com/raoyongming/CAL/blob/master/fgvc/utils.py
https://github.com/raoyongming/CAL/blob/master/fgvc/models/cal.py
https://github.com/raoyongming/CAL/blob/master/fgvc/models/resnet.py
https://github.com/raoyongming/CAL/blob/master/fgvc/models/inception.py
https://github.com/raoyongming/CAL/blob/master/fgvc/infer.py
'''
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import torch.utils.model_zoo as model_zoo
from einops import rearrange


EPSILON = 1e-6


def get_cal_config():
    """Returns the CAL configuration."""
    config = ml_collections.ConfigDict()
    config.net = 'resnet101'
    config.num_attentions = 32
    config.beta = 5e-2
    # config.opt_level = o0 # fp 32
    # config.epochs = 160
    # config.batch_size = 4 * 4 gpus = 16
    # configs.learning_rate = 1e-3
    # config.weight_decay = 1e-5
    return config


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], \
            but received unsupported augmentation method %s' % mode)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, cbam=None, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cbam=None, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, cbam=None, num_classes=1000, stride=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], cbam, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cbam, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cbam, stride=stride)
        print('==> using resnet with stride=', 16 * stride)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cbam=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, cbam=cbam, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cbam=cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_features(self):
        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        model_dict.update(pretrained_dict)
        super(ResNet, self).load_state_dict(model_dict)


def resnet101(pretrained=False, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
    return model


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class WSDAN_CAL(nn.Module):
    """
    WS-DAN models
    Hu et al.,
    "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network
    for Fine-Grained Visual Classification",
    arXiv:1901.09891
    """
    def __init__(self, num_classes, M=32, net='resnet101', pretrained=False):
        super(WSDAN_CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if net == 'resnet101':
            self.features = resnet101(pretrained=True).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        print('WSDAN: using {} as feature extractor, num_classes: {}, \
              num_attentions: {}'.format(net, self.num_classes, self.M))

    def visualize(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        attention_maps = self.attentions(feature_maps)

        feature_matrix = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)

        return p, attention_maps

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        attention_maps = self.attentions(feature_maps)

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN_CAL, self).load_state_dict(model_dict)


# Center Loss for Attention Regularization
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


class CAL(nn.Module):
    def __init__(self, config, num_classes=21843, device='cuda', single_crop=False):
        super(CAL, self).__init__()
        self.config = config
        self.net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions,
                             net=config.net, pretrained=True)
        self.feature_center = torch.zeros(
            num_classes, config.num_attentions * self.net.num_features, device=device)
        self.center_loss = CenterLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.single_crop = single_crop
        self.no_crops = False

    def forward(self, x, y=None, train=False):
        if train:
            # raw image
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.net(x)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.config.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
            aug_images = torch.cat([crop_images, drop_images], dim=0)
            y_aug = torch.cat([y, y], dim=0)

            # crop images forward
            y_pred_aug, y_pred_aux_aug, _, _ = self.net(aug_images)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)

            batch_loss = (self.cross_entropy_loss(y_pred_raw, y) / 3. +
                          self.cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. +
                          self.cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. +
                          self.center_loss(feature_matrix, feature_center_batch))

            # final prediction
            y_pred_aug, _ = torch.split(y_pred_aug, [x.shape[0], x.shape[0]], dim=0)
            y_pred = (y_pred_raw + y_pred_aug) / 2.
            return y_pred, batch_loss, crop_images
        elif self.single_crop:
            y_pred_raw, y_pred_aux, _, attention_map = self.net(x)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, _, _, _ = self.net(crop_images3)

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop3) / 2.

            if y is None:
                return y_pred, crop_images3

            batch_loss = self.cross_entropy_loss(y_pred, y)
            return y_pred, batch_loss, crop_images3

        elif self.no_crops:
            y_pred_raw, y_pred_aux, _, attention_map = self.net(x)
            return y_pred_raw

        else:
            x_m = torch.flip(x, [3])
            # Raw Image
            y_pred_raw, _, _, attention_map = self.net(x)
            y_pred_raw_m, _, _, attention_map_m = self.net(x_m)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
            y_pred_crop, _, _, _ = self.net(crop_images)

            crop_images2 = batch_augment(x, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
            y_pred_crop2, _, _, _ = self.net(crop_images2)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, _, _, _ = self.net(crop_images3)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
            y_pred_crop_m, _, _, _ = self.net(crop_images_m)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
            y_pred_crop_m2, _, _, _ = self.net(crop_images_m2)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop_m3, _, _, _ = self.net(crop_images_m3)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            if y is None:
                crop_images = torch.stack((crop_images, crop_images2, crop_images3,
                                           crop_images_m, crop_images_m2, crop_images_m3), dim=-1)
                return y_pred, crop_images

            batch_loss = self.cross_entropy_loss(y_pred, y)
            return y_pred, batch_loss, crop_images


CONFIGS_CAL = {
    'cal': get_cal_config()
}
