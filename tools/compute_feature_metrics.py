import os
import random
import argparse
import json
from warnings import warn
from typing import List, Dict
from pathlib import Path
from functools import partial
from contextlib import suppress
from statistics import mean, stdev

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange
from timm.models import create_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from glsim.data_utils.build_dataloaders import build_dataloaders
from glsim.other_utils.build_args import parse_inference_args
from glsim.model_utils.build_model import build_model
from glsim.train_utils.misc_utils import set_random_seed


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


MODELS_DIC = {
    'vit_b16': 'ViT',
    'transfg': 'TransFG',
    'ffvt': 'FFVT',
    'resnet101': 'ResNet101',
    'cal': 'CAL',
    'glsim_b16': 'GLSim',
}

DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'moe': 'Moe',
    'pets': 'Pets',
    'soy': 'SoyLocal',
}

SETTINGS_DIC = {
    'ft_224_strong': 'IS=224 & Strong AugReg',
    'ft_224_minimal': 'IS=224 & Minimal AugReg',
    'ft_224_weak': 'IS=224 & Weak AugReg',
    'ft_224_medium': 'IS=224 & Medium AugReg',
    'ft_448_strong': 'IS=448 & Strong AugReg',
    'ft_448_minimal': 'IS=448 & Minimal AugReg',
    'ft_448_weak': 'IS=448 & Weak AugReg',
    'ft_448_medium': 'IS=448 & Medium AugReg',
    'fz_strong': 'Frozen & Strong AugReg',
    'fz_minimal': 'Frozen & Minimal AugReg',
}


def adjust_args_general(args, setting):
    if args.anchor_size:
        args.run_name = '{}_{}_{}_{}_{}'.format(
            args.dataset_name, args.model_name, args.anchor_size, setting, args.serial
        )
    else:
        args.run_name = '{}_{}_{}_{}'.format(
            args.dataset_name, args.model_name, setting, args.serial
        )

    args.results_dir = os.path.join(args.results_inference, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)

    return args


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

class FeatureMetrics:
    def __init__(self,
                 model: nn.Module,
                 model_name: str = None,
                 model_layers: List[str] = None,
                 device: str ='cpu',
                 image_size: int = 224,
                 setting: str = 'fz',
                 out_size: int = 7,
                 debugging: bool = False):
        """

        :param model: (nn.Module) Neural Network 1
        :param model_name: (str) Name of model 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Setting'] = SETTINGS_DIC.get(setting, setting)

        if model_name is None:
            self.model_info['Name_og'] = model.__repr__().split('(')[0]
        else:
            self.model_info['Name_og'] = model_name
        self.model_info['Name'] = MODELS_DIC.get(self.model_info['Name_og'], self.model_info['Name_og'])

        self.model_info['Layers'] = []

        self.model_features = {}

        if len(list(model.modules())) > 150 and model_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model_layers' parameter. Your CPU/GPU will thank you :)")

        self.model_layers = model_layers

        self._insert_hooks()
        self.model = self.model.to(self.device)

        self.model.eval()

        self._check_shape(image_size)

        self.pool = nn.AdaptiveAvgPool2d((out_size, out_size)).to(self.device)

        self.debugging = debugging

        print(self.model_info)

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))

    def _check_shape(self, image_size):
        with torch.no_grad():
            x = torch.rand(2, 3, image_size, image_size).to(self.device)
            _ = self.model(x)

            # -1 in certain cases corresponds to classification layer
            last = self.model_info['Layers'][-2]
            feat_out = self.model_features[last]

            if len(feat_out.shape) == 4:
                b, c, h, w = feat_out.shape
                if h == w:
                    self.bchw = True
                    h = feat_out.shape[-1]
                else:
                    self.bchw = False
                    h = feat_out.shape[1]
            elif len(feat_out.shape) == 3:
                h = int(feat_out.shape[1] ** 0.5)
                self.cls = False if h ** 2 == feat_out.shape[1] else True
            else:
                pass

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def _pool_features(self, feat, pool=True):
        if hasattr(self, 'pool') and pool:
            if len(feat.shape) == 4:
                if not self.bchw:
                    feat = rearrange(feat, 'b h w c -> b c h w')
                pooled = self.pool(feat)
                pooled = rearrange(pooled, 'b c h w -> (b h w) c')
            elif len(feat.shape) == 3:
                h = int(feat.shape[1] ** 0.5)
                if self.cls:
                    x_cls, x_others = torch.split(feat, [1, int(h**2)], dim=1)
                    x_others = rearrange(x_others, 'b (h w) d -> b d h w', h=h)
                    x_others = self.pool(x_others)
                    x_others = rearrange(x_others, 'b d h w -> b (h w) d')
                    pooled = torch.cat([x_cls, x_others], dim=1)
                    pooled = rearrange(pooled, 'b s d -> (b s) d')
                else:
                    pooled = rearrange(feat, 'b (h w) d -> b d h w', h=h)
                    pooled = self.pool(pooled)
                    pooled = rearrange(pooled, 'b c h w -> (b h w) c')                    
        else:
            pooled = feat.flatten(1)
 
        return pooled

    def compare(self,
                dataloader1: DataLoader) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        """

        self.model_info['Dataset_og'] = dataloader1.dataset.dataset_name
        self.model_info['Dataset'] = DATASETS_DIC.get(self.model_info['Dataset_og'], self.model_info['Dataset_og'])

        N = len(self.model_layers) if self.model_layers is not None else len(list(self.model.modules()))

        self.hsic_matrix = torch.zeros(N, N, 3)
        self.hsic_matrix_pooled = torch.zeros(N, N, 3)
        self.dist_cum = torch.zeros(N, device=self.device)
        self.dist_cum_norm = torch.zeros(N, device=self.device)
        self.l2_norm = torch.zeros(N, device=self.device)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches):

            self.model_features = {}
            x1 = x1.to(self.device)
            _ = self.model(x1)

            for i, (name1, feat1) in enumerate(self.model_features.items()):
                X = self._pool_features(feat1, pool=False)
                X_pooled = self._pool_features(feat1, pool=True)

                # frobenius norm
                self.l2_norm[i] += torch.norm(X_pooled, p='fro', dim=-1).mean() / num_batches

                dist = torch.cdist(X_pooled, X_pooled, p=2.0)

                dist_avg = (torch.sum(dist) / torch.nonzero(dist).size(0))
                self.dist_cum[i] += dist_avg / num_batches

                dist = (dist - dist.min()) / (dist.max() - dist.min())
                dist_avg_norm = (torch.sum(dist) / torch.nonzero(dist).size(0))
                self.dist_cum_norm[i] += dist_avg_norm / num_batches

                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                K_pooled = X_pooled @ X_pooled.t()
                K_pooled.fill_diagonal_(0.0)
                self.hsic_matrix_pooled[i, :, 0] += self._HSIC(K_pooled, K_pooled) / num_batches

                for j, (name2, feat2) in enumerate(self.model_features.items()):
                    Y = self._pool_features(feat2, pool=False)
                    Y_pooled = self._pool_features(feat2, pool=True)

                    L = Y @ Y.t()
                    L.fill_diagonal_(0)

                    L_pooled = Y_pooled @ Y_pooled.t()
                    L_pooled.fill_diagonal_(0)

                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                    assert K_pooled.shape == L_pooled.shape, f"Feature shape mistach! {K_pooled.shape}, {L_pooled.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

                    self.hsic_matrix_pooled[i, j, 1] += self._HSIC(K_pooled, L_pooled) / num_batches
                    self.hsic_matrix_pooled[i, j, 2] += self._HSIC(L_pooled, L_pooled) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())
        self.hsic_matrix_pooled = self.hsic_matrix_pooled[:, :, 1] / (self.hsic_matrix_pooled[:, :, 0].sqrt() *
                                                        self.hsic_matrix_pooled[:, :, 2].sqrt())


    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model_name": self.model_info['Name'],
            "model_name_og": self.model_info['Name_og'],
            "model_layers": self.model_info['Layers'],
            "dataset_name": self.model_info['Dataset'],
            "dataset_name_og": self.model_info['Dataset_og'],
            "setting": self.model_info['Setting'],
            'l2_norm': self.l2_norm,
            "CKA": self.hsic_matrix,
            "CKA_pooled": self.hsic_matrix_pooled,
            "dist": self.dist_cum,
            "dist_norm": self.dist_cum_norm,
        }

    def plot_cka(self,
                 save_path: str = None,
                 title: str = None,
                 show: bool = False):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')

        ax.set_xlabel(f"Layers", fontsize=12)
        ax.set_ylabel(f"Layers", fontsize=12)

        labels = range(self.hsic_matrix.shape[0])
        ax.set_xticks(labels)
        ax.set_yticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=13)
        else:
            title = f"CKA for {self.model_info['Name']} on {self.model_info['Dataset']}\n w/ {self.model_info['Setting']}"
            ax.set_title(title, fontsize=13)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()

    def plot_metrics(self,
                     metric: str = 'norms',
                     save_path: str = None,
                     title: str = None,
                     show: bool = False):
        fig, ax = plt.subplots()

        if metric == 'norms':
            labels = range(self.l2_norm.shape[0])
            ax.bar(labels, self.l2_norm.cpu())
            y_label = 'L2-Norm'
        elif metric == 'dist':
            labels = range(self.dist_cum.shape[0])
            ax.bar(labels, self.dist_cum.cpu())
            y_label = 'L2-Distance'
        elif metric == 'dist_norm':
            labels = range(self.dist_cum_norm.shape[0])
            ax.bar(labels, self.dist_cum_norm.cpu())
            y_label = 'Normalized L2-Distance'

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=13)
        else:
            title = f"Average {y_label} for {self.model_info['Name']} on {self.model_info['Dataset']}\n w/ {self.model_info['Setting']}"
            ax.set_title(title, fontsize=13)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()


def calc_cka(results, pooled=False, split='train'):
    name = 'CKA_pooled' if pooled else 'CKA'

    ckas = {}

    results = results[name]

    for i, cka in enumerate(results):
        # 1st order derivative of cka with respect to layers
        cka = cka.tolist()
        layer_change = [abs(cka[l + 1] - cka[l]) for l in range(0, len(cka) - 1)]
        layer_change_mean = mean(layer_change)
        layer_change_std = stdev(layer_change)

        # 2nd order derivative of cka with respect to layers (change of layer change)
        layer_change_2nd = [abs(layer_change[i + 1] - layer_change[i]) for i in range(0, len(layer_change) - 1)]
        layer_change_2nd_mean = mean(layer_change_2nd)
        layer_change_2nd_std = stdev(layer_change_2nd)

        ckas.update({
            f'{name.lower()}_change_mean_{i}_{split}': layer_change_mean,
            f'{name.lower()}_change_std_{i}_{split}': layer_change_std,
            f'{name.lower()}_change2_mean_{i}_{split}': layer_change_2nd_mean,
            f'{name.lower()}_change2_std_{i}_{split}': layer_change_2nd_std,
            })

    results = results.fill_diagonal_(0)

    for i, cka in enumerate(results):
        layer_mean = (torch.sum(cka) / torch.nonzero(cka).size(0)).item()
        ckas.update({f'{name.lower()}_{i}_{split}': layer_mean})

    overall_mean = (torch.sum(results) / torch.nonzero(results).size(0)).item()
    ckas.update({f'{name.lower()}_avg_{split}': overall_mean})

    return ckas


def calc_distances(results, split='train'):
    dists = {}
    for i, (dist, dist_norm) in enumerate(zip(results['dist'], results['dist_norm'])):
        dists.update({f'dist_{i}_{split}': dist.item(), f'dist_norm_{i}_{split}': dist_norm.item()})

    dists.update({f'dist_avg_{split}': torch.mean(results['dist']).item(),
                  f'dist_norm_avg_{split}': torch.mean(results['dist_norm']).item()})
    return dists


def calc_l2_norm(results, split='train'):
    norms = {}
    for i, norm in enumerate(results['l2_norm']):
        norms.update({f'l2_norm_{i}_{split}': norm.item()})

    norms.update({f'l2_norm_avg_{split}': torch.mean(results['l2_norm']).item()})
    return norms



def save_results_to_json(args, results_train, results_test):
    # needs to convert tensors (l2_norm, dist, dist_norm, CKA, CKA_pooled) to list
    results_train['l2_norm'] = results_train['l2_norm'].tolist()
    results_train['dist'] = results_train['dist'].tolist()
    results_train['dist_norm'] = results_train['dist_norm'].tolist()
    results_train['CKA'] = results_train['CKA'].tolist()
    results_train['CKA_pooled'] = results_train['CKA_pooled'].tolist()

    results_test['l2_norm'] = results_test['l2_norm'].tolist()
    results_test['dist'] = results_test['dist'].tolist()
    results_test['dist_norm'] = results_test['dist_norm'].tolist()
    results_test['CKA'] = results_test['CKA'].tolist()
    results_test['CKA_pooled'] = results_test['CKA_pooled'].tolist()

    data = {'train': results_train, 'test': results_test} 

    fp = os.path.join(args.results_dir, 'feature_metrics.json')
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)

    return 0


def setup_environment(args):
    set_random_seed(args.seed, numpy=True)

    # dataloaders
    args.shuffle_test = True
    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = build_model(args)

    if args.ckpt_path:
        setting = 'ft'

        ds_mn_serial = args.ckpt_path.split('/')[-1]
        serial = ds_mn_serial.split('_')[-1].replace('.pth', '')

        if int(serial) == 2:
            setting = 'ft_224_strong'
        elif int(serial) == 3:
            setting = 'ft_448_strong'
        elif int(serial) == 4:
            setting = 'ft_448_minimal'
        elif int(serial) == 5:
            setting = 'ft_448_weak'
        elif int(serial) == 6:
            setting = 'ft_448_medium'
        elif int(serial) == 7:
            setting = 'ft_224_minimal'
        elif int(serial) == 8:
            setting = 'ft_224_weak'
        elif int(serial) == 9:
            setting = 'ft_224_medium'
    else:
        if args.trivial_aug and args.re:
            setting = 'fz_strong'
        else:
            setting = 'fz_minimal'

    args = adjust_args_general(args, setting)

    if not args.debugging:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name

    if args.model_name == 'resnet101': 
        layers = ['model.encoder.layer1.2.bn3', 'model.encoder.layer2.3.bn3', 'model.encoder.layer3.22.bn3', 'model.encoder.layer4.2.bn3']
    elif args.model_name == 'cal':
        layers = ['model.net.features.4.2.bn3', 'model.net.features.5.3.bn3', 'model.net.features.6.22.bn3', 'model.net.features.7.2.bn3']
        model.model.no_crops = True
    elif 'vit_b' in args.model_name:
        layers = ['model.encoder.blocks.2.norm2', 'model.encoder.blocks.5.norm2', 'model.encoder.blocks.8.norm2', 'model.encoder.blocks.11.norm2']
        # layers = ['model.encoder.blocks.2.norm2', 'model.encoder.blocks.5.norm2', 'model.encoder.blocks.8.norm2', 'model.encoder.blocks.11.norm2', 'model.encoder_norm']
        if args.anchor_size:
            args.model_name = args.model_name.replace('vit', 'glsim')
            delattr(model.model, 'anchor_size')
            delattr(model.model, 'aggregator')
            delattr(model.model, 'reducer')
        if hasattr(model.model, 'head_aux'):
            delattr(model.model, 'head_aux')
    elif args.model_name == 'transfg':
        layers = ['model.transformer.encoder.layer.2.ffn_norm', 'model.transformer.encoder.layer.5.ffn_norm', 'model.transformer.encoder.layer.8.ffn_norm', 'model.transformer.encoder.layer.10.ffn_norm']
        # layers = ['model.transformer.encoder.layer.2.ffn_norm', 'model.transformer.encoder.layer.5.ffn_norm', 'model.transformer.encoder.layer.8.ffn_norm', 'model.transformer.encoder.part_layer.ffn_norm', 'model.transformer.encoder.part_norm']
    elif args.model_name == 'ffvt':
        layers = ['model.transformer.encoder.layer.2.ffn_norm', 'model.transformer.encoder.layer.5.ffn_norm', 'model.transformer.encoder.layer.8.ffn_norm', 'model.transformer.encoder.layer.10.ffn_norm']
        # layers = ['model.transformer.encoder.layer.2.ffn_norm', 'model.transformer.encoder.layer.5.ffn_norm', 'model.transformer.encoder.layer.8.ffn_norm', 'model.transformer.encoder.ff_last_layer.ffn_norm', 'model.transformer.encoder.ff_encoder_norm']
    else:
        raise NotImplementedError

    feature_metrics = FeatureMetrics(model, args.model_name, layers, args.device,
                          args.image_size, setting, debugging=args.debugging)

    return train_loader, test_loader, feature_metrics, setting


def main():
    args = parse_inference_args()

    train_loader, test_loader, feature_metrics, setting = setup_environment(args)

    amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress

    with torch.no_grad():
        with amp_autocast():
            feature_metrics.compare(train_loader)

            results_train = feature_metrics.export()
            feature_metrics.plot_cka(os.path.join(args.results_dir, 'cka_train.png'))
            feature_metrics.plot_metrics('norms', os.path.join(args.results_dir, 'norms_train.png'))
            feature_metrics.plot_metrics('dist', os.path.join(args.results_dir, 'dist_train.png'))
            feature_metrics.plot_metrics('dist_norm', os.path.join(args.results_dir, 'dist_norm_train.png'))

            cka_train = calc_cka(results_train, split='train')
            cka_pooled_train = calc_cka(results_train, pooled=True, split='train')
            dists_train = calc_distances(results_train, split='train')
            norms_train = calc_l2_norm(results_train, split='train')

            feature_metrics.compare(test_loader)

            results_test = feature_metrics.export()
            feature_metrics.plot_cka(os.path.join(args.results_dir, 'cka_test.png'))
            feature_metrics.plot_metrics('norms', os.path.join(args.results_dir, 'norms_test.png'))
            feature_metrics.plot_metrics('dist', os.path.join(args.results_dir, 'dist_test.png'))
            feature_metrics.plot_metrics('dist_norm', os.path.join(args.results_dir, 'dist_norm_test.png'))

            cka_test = calc_cka(results_test, split='test')
            cka_pooled_test = calc_cka(results_test, pooled=True, split='test')
            dists_test = calc_distances(results_test, split='test')
            norms_test = calc_l2_norm(results_test, split='test')

    log_dic = {'setting': setting}
    log_dic.update(cka_train)
    log_dic.update(cka_pooled_train)
    log_dic.update(dists_train)
    log_dic.update(norms_train)
    log_dic.update(cka_test)
    log_dic.update(cka_pooled_test)
    log_dic.update(dists_test)
    log_dic.update(norms_test)

    if not args.debugging:
        wandb.log(log_dic)
        wandb.finish()
    else:
        print(log_dic)

    save_results_to_json(args, results_train, results_test)

    return 0


if __name__ == '__main__':
    main()