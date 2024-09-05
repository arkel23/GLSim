""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import logging
import math
from random import randint
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from einops.layers.torch import Rearrange
from einops import repeat, reduce, rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model


__all__ = ['TIMMGLSViT']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, adapter, dim, adapter_dim, kernel_size):
        super().__init__()

        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, dim)
        self.act = nn.GELU()

        if adapter == 'convpass':
            padding = kernel_size // 2
            # conv with kernel size 3, stride 1, padding=1
            self.adapter_conv = nn.Conv2d(adapter_dim, adapter_dim, kernel_size, 1, padding)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if hasattr(self, 'adapter_down'):
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.xavier_uniform_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)
        if hasattr(self, 'adapter_conv'):
            nn.init.xavier_uniform_(self.adapter_conv.weight)

    def forward(self, x):
        x = self.act(self.adapter_down(x))

        if hasattr(self, 'adapter_conv'):
            _, s, _ = x.shape
            h = int(s ** 0.5)
            cls = False if (h ** 2) == s else True

            if cls:
                x_cls, x_patches = torch.split(x, [1, s - 1], dim=1)
                x_cls = rearrange(x_cls, 'b 1 d -> b d 1 1')
                x_cls = self.adapter_conv(x_cls)
                x_cls = rearrange(x_cls, 'b d 1 1 -> b 1 d')

                x_patches = rearrange(x_patches, 'b (h w) d -> b d h w', h=h)
                x_patches = self.adapter_conv(x_patches)
                x_patches = rearrange(x_patches, 'b d h w -> b (h w) d')

                x = torch.cat([x_cls, x_patches], dim=1)
                x = self.act(x)
            else:
                x = rearrange(x, 'b (h w) d -> b d h w', h=h)
                x = self.adapter_conv(x)
                x = rearrange(x, 'b d h w -> b (h w) d')                
                x = self.act(x)

        x = self.adapter_up(x)
        return x


class GLSimCrop(nn.Module):
    def __init__(self, dynamic_top=8, patch_equivalent=16, sim_metric='cos',
                 cls_token=True, select_top_k=False, debugging=False,):
        super().__init__()
        self.dynamic_top = dynamic_top
        self.patch_equivalent = patch_equivalent
        self.sim_metric = sim_metric

        if cls_token:
            self.class_token = True

        if select_top_k:
            self.select_top_k = True

        self.debugging = debugging


    def get_glsim_indices(self, x):
        if hasattr(self, 'class_token'):
            g = x[:, :1, :]
            l = x[:, 1:, :]
        else:
            g = reduce(x, 'b s d -> b d', 'mean')
            l = x

        if self.sim_metric in ('l1'):
            distances = F.l1_loss(g, l, reduction='none')
        elif self.sim_metric in ('l2'):
            distances = F.mse_loss(g, l, reduction='none')
        elif self.sim_metric in ('cos'):
            distances = F.cosine_similarity(g, l, dim=-1)

        if self.sim_metric != 'cos':
            distances = reduce(distances, 'b a k -> b a', 'mean')
            distances = torch.abs(distances)

        largest = True if self.sim_metric == 'cos' else False

        k = self.dynamic_top

        # k = self.dynamic_top
        top, ind = distances.topk(k, dim=-1, largest=largest)
        self.maybe_print('Distances and 1-d indexes: ', distances.shape, top.shape,
                         ind.shape, top[0], ind[0])

        if hasattr(self, 'select_top_k'):
            ind = repeat(ind, 'b i -> b i d', d=l.shape[-1])
            selected = torch.gather(l, dim=1, index=ind)
            return selected, distances, top, ind

        avg = reduce(distances, 'b s -> b', 'mean')
        stds = torch.std(distances, dim=-1)

        return distances, top, ind, avg, stds

    @torch.no_grad()
    def crop_image(self, images, i, x_0, x_1, y_0, y_1):
        crop = images[i:i+1, :, x_0:x_1, y_0:y_1]
        crop = F.upsample_bilinear(crop, size=(images.shape[-1], images.shape[-1]))
        return crop

    @torch.no_grad()
    def forward(self, x, images):
        self.maybe_print(f'Obtaining crops: ', self.training, x.shape, images.shape)
        if isinstance(images.shape[-1], torch.Tensor):
            img_size = images.shape[-1].item()
        else:
            img_size = images.shape[-1]

        if hasattr(self, 'select_top_k'):
            selected, distances, top, ind = self.get_glsim_indices(x)
            return selected
        else:
            distances, top, ind, avgs, stds = self.get_glsim_indices(x)

        fh = int(math.sqrt(x.shape[1]))

        ind_x = torch.floor(torch.div(ind, fh)).int()
        ind_y = ind % fh

        ind_x_i = reduce(ind_x, 'b a -> b', 'min')
        ind_x_f = reduce(ind_x, 'b a -> b', 'max')
        ind_y_i = reduce(ind_y, 'b a -> b', 'min')
        ind_y_f = reduce(ind_y, 'b a -> b', 'max')

        x_i = ind_x_i * self.patch_equivalent
        y_i = ind_y_i * self.patch_equivalent
        x_f = ind_x_f * self.patch_equivalent
        y_f = ind_y_f * self.patch_equivalent

        self.maybe_print('2d indexes: ', ind_x[0], ind_y[0], x_i, x_f, y_i, y_f)

        images_crops = []
        for i in range(ind.shape[0]):
            x_0 = max(x_i[i], 0)
            y_0 = max(y_i[i], 0)
            x_1 = min(max(x_f[i], x_i[i] + self.patch_equivalent), img_size)
            y_1 = min(max(y_f[i], y_i[i] + self.patch_equivalent), img_size)
            aug = self.crop_image(images, i, x_0, x_1, y_0, y_1)

            images_crops.append(aug)

        images_crops = torch.cat(images_crops, dim=0)

        return images_crops

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            token_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.pre_softmax = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if token_drop:
            self.token_drop = nn.Dropout1d(token_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        if hasattr(self, 'token_drop'):
            x_cls, x_others = torch.split(x, [1, N-1], dim=1)
            x_others = self.token_drop(x_others)
            x = torch.cat([x_cls, x_others], dim=1)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = self.pre_softmax(attn)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            adapter: str = None,
            token_drop: float = 0.,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            token_drop=token_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if adapter:
            self.adapter_attn = Adapter(adapter, dim, int(dim//96), 3)
            self.adapter_pwffn = Adapter(adapter, dim, int(dim//96), 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        res =  x

        x = self.norm1(x)
        h = self.drop_path1(self.ls1(self.attn(x)))

        if hasattr(self, 'adapter_attn'):
            h2 = self.drop_path1(self.adapter_attn(x))
            x = res + h + h2
        else:
            x = res + h


        res =  x
        x = self.norm2(x)
        h = self.drop_path2(self.ls2(self.mlp(x)))

        if hasattr(self, 'adapter_pwffn'):
            h2 = self.drop_path1(self.adapter_pwffn(x))
            x = res + h + h2
        else:
            x = res + h

        return x


class TIMMGLSViT(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.0,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            args=None,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        adapter = getattr(args, 'adapter', None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                adapter=adapter,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.debugging = getattr(args, 'debugging', True)
        dynamic_anchor = getattr(args, 'dynamic_anchor', True)
        if dynamic_anchor:
            # cropping module
            dynamic_top = getattr(args, 'dynamic_top', 8)
            sim_metric = getattr(args, 'sim_metric', 'cos')
            select_top_k = getattr(args, 'select_top_k', False)
            if select_top_k:
                self.select_top_k = GLSimCrop(dynamic_top, patch_size, sim_metric,
                                            class_token, select_top_k, self.debugging)
            else:
                self.inference_crops = getattr(args, 'inference_crops', True)
                self.get_crops = GLSimCrop(dynamic_top, patch_size, sim_metric,
                                            class_token, debugging=self.debugging)

            aggregator_type = getattr(args, 'aggregator_type', 'transformer')

            if aggregator_type == 'transformer':
                self.aggregator = nn.Sequential(
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        init_values=init_values,
                        proj_drop=proj_drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rate,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        mlp_layer=mlp_layer,
                        token_drop=getattr(args, 'token_drop', 0.2),
                    ),
                    nn.LayerNorm(embed_dim)
                )
            elif aggregator_type == 'dwsc':
                self.aggregator = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    Rearrange('b s d -> b d s'),
                    nn.Conv1d(embed_dim, embed_dim, 1+dynamic_top, groups=embed_dim),
                    # nn.BatchNorm1d(embed_dim),
                    Rearrange('b d s -> b s d'),
                    nn.LayerNorm(embed_dim),
                    Rearrange('b s d -> b d s'),
                    nn.GELU(),
                    nn.Conv1d(embed_dim, embed_dim, 1),
                    Rearrange('b d s -> b s d'),
                    nn.LayerNorm(embed_dim),
                )
            elif aggregator_type == 'conv':
                self.aggregator = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    Rearrange('b s d -> b d s'),
                    nn.Conv1d(embed_dim, embed_dim, 1+dynamic_top),
                    Rearrange('b d s -> b s d'),
                    nn.LayerNorm(embed_dim),
                )
            else:
                raise NotImplementedError

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', ''] = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token', 'cls_token_crops'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map', 'pool', 'cls_pool')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool

        if global_pool == 'cls_pool':
            self.head = nn.Linear(self.embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        return 0

    def add_crop_cls(self) -> None:
        self.cls_token_crops = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token_crops, std=1e-6)
        with torch.no_grad():
            cls_token_weights = self.cls_token
            self.cls_token_crops.copy_(cls_token_weights)
            print(f'Loaded crops CLS token from CLS token')

        if hasattr(self, 'aggregator'):
            self.norm = nn.Identity()

        return 0

    def _pos_embed(self, x: torch.Tensor, crops=False) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if crops and hasattr(self, 'cls_token_crops'):
            to_cat.append(self.cls_token_crops.expand(x.shape[0], -1, -1))
        elif self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def get_new_module_names(self):
        if hasattr(self, 'aggregator'):
            return ['aggregator']
        return []

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        b, s, d = x.shape

        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for i, blck in enumerate(self.blocks):
            x = blck(x)
            self.maybe_print('Encoder blocks: ', i, x.shape)

        if self.reg_token is not None:
            x = torch.cat([x[:, :1], x[:, -s:]], dim=1)
        x = self.norm(x)

        if hasattr(self, 'select_top_k'):
            # get top k tokens
            top_k = self.select_top_k(x, images)
            x = torch.cat([x[:, :1], top_k], dim=1)
            self.maybe_print('After selection: ', x.shape)

            # aggregator
            x = self.aggregator(x)
            self.maybe_print('After aggregator: ', x.shape)

        elif hasattr(self, 'get_crops') and (self.inference_crops or self.training):
            # get crops
            crops = self.get_crops(x, images)

            # crops forward pass
            x_crops = self.patch_embed(crops)

            x_crops = self._pos_embed(x_crops, crops=True)
            x_crops = self.patch_drop(x_crops)
            x_crops = self.norm_pre(x_crops)

            for i, blck in enumerate(self.blocks):
                x_crops = blck(x_crops)
                self.maybe_print('Encoder blocks for crops: ', i, x_crops.shape)

            x_crops = self.norm(x_crops)

            # reducer
            x = x[:, :1, :]
            x_crops = x_crops[:, :1, :]
            x = torch.cat([x, x_crops], dim=1)

            # aggregator
            x = self.aggregator(x)
            self.maybe_print('After aggregator: ', x.shape)

        if hasattr(self, 'get_crops'):
            x = x[:, 0]
        elif self.global_pool == 'cls_pool':
            x_cls, x_others = x[:, 0], x[:, 1:]
            x_others = x_others.mean(dim=1)
            x = torch.cat([x_cls, x_others], dim=-1)
        elif self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool == 'pool':
            x = x.mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)

        self.maybe_print('After classifier: ', x.shape)
        if hasattr(self, 'get_crops'):
            crops = torch.cat([images, crops], dim=0)
            return x, crops
        return x

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> None:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: TIMMGLSViT, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: TIMMGLSViT,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: TIMMGLSViT,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: TIMMGLSViT,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'module.cls_token' in state_dict:
        # mocov3
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

default_cfgs = {
    # patch models, imagenet21k (weights from official Google JAX impl), classifier not valid
    'glsvit_base_patch32_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        hf_hub_id='timm/vit_base_patch32_224.orig_in21k',
        num_classes=0),
    'glsvit_base_patch16_224.orig_in21k': _cfg(
        # url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        hf_hub_id='timm/vit_base_patch16_224.orig_in21k',
        num_classes=0),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'glsvit_small_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        hf_hub_id='timm/vit_small_patch16_224.dino',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'glsvit_base_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        hf_hub_id='timm/vit_base_patch16_224.dino',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    # DINOv2 pretrained - https://arxiv.org/abs/2304.07193 (no classifier head, for fine-tune/features only)
    'glsvit_small_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        hf_hub_id='timm/vit_small_patch14_dinov2.lvd142m',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'glsvit_base_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        hf_hub_id='timm/vit_base_patch14_dinov2.lvd142m',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'glsvit_large_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        hf_hub_id='timm/vit_large_patch14_dinov2.lvd142m',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # DINOv2 pretrained w/ registers - https://arxiv.org/abs/2309.16588 (no classifier head, for fine-tune/features only)
    'glsvit_small_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth',
        hf_hub_id='timm/vit_small_patch14_reg4_dinov2.lvd142m',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'glsvit_base_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
        hf_hub_id='timm/vit_base_patch14_reg4_dinov2.lvd142m',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # ViT ImageNet-21K-P pretraining by MILL
    'glsvit_base_patch16_224_miil.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_in21k_miil-887286df.pth',
        hf_hub_id='timm/vit_base_patch16_224_miil.in21k',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221),

    # CLIP pretrained image tower and related fine-tuned weights
    'glsvit_base_patch16_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'glsvit_base_patch16_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),

    # SSL
    'glsvit_base_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        hf_hub_id='timm/vit_base_patch16_224.mae',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'glsvit_large_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        hf_hub_id='timm/vit_large_patch16_224.mae',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'glsvit_huge_patch14_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        hf_hub_id='timm/vit_huge_patch14_224.mae',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'glsvit_huge_patch14_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'glsvit_small_patch16_224.in1k_mocov3': _cfg(
        url='https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/linear-vit-s-300ep.pth.tar',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=1000),
    'glsvit_base_patch16_224.in1k_mocov3': _cfg(
        url='https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=1000),

    # deit models (FB weights)
    'glsdeit_tiny_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit_tiny_patch16_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'glsdeit_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit_small_patch16_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'glsdeit_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit_base_patch16_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),

    'glsdeit3_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit3_small_patch16_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'),
    'glsdeit3_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit3_base_patch16_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'),
    'glsdeit3_large_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit3_large_patch16_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth'),
    'glsdeit3_huge_patch14_224.fb_in1k': _cfg(
        hf_hub_id='timm/deit3_huge_patch14_224.fb_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth'),

    'glsdeit3_small_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/deit3_small_patch16_224.fb_in22k_ft_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
        crop_pct=1.0),
    'glsdeit3_base_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/deit3_base_patch16_224.fb_in22k_ft_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
        crop_pct=1.0),
    'glsdeit3_large_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/deit3_large_patch16_224.fb_in22k_ft_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
        crop_pct=1.0),
    'glsdeit3_huge_patch14_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/deit3_huge_patch14_224.fb_in22k_ft_in1k',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
}

default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = False
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        TIMMGLSViT,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


@register_model
def glsvit_tiny_patch16_224(pretrained: bool = False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('glsvit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_small_patch16_224(pretrained: bool = False, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('glsvit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_small_patch8_224(pretrained: bool = False, **kwargs):
    """ ViT-Small (ViT-S/8)
    """
    model_args = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('glsvit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def glsvit_base_patch32_224(pretrained: bool = False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('glsvit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def glsvit_base_patch16_224(pretrained: bool = False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('glsvit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch8_224(pretrained: bool = False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('glsvit_base_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_large_patch16_224(pretrained: bool = False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('glsvit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_large_patch14_224(pretrained: bool = False, **kwargs):
    """ ViT-Large model (ViT-L/14)
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('glsvit_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_huge_patch14_224(pretrained: bool = False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('glsvit_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_224_miil(pretrained: bool = False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
    model = _create_vision_transformer(
        'glsvit_base_patch16_224_miil', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_medium_patch16_gap_240(pretrained: bool = False, **kwargs):
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 240x240
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'glsvit_medium_patch16_gap_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_medium_patch16_gap_256(pretrained: bool = False, **kwargs):
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 256x256
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'glsvit_medium_patch16_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_gap_224(pretrained: bool = False, **kwargs):
    """ ViT-Base (ViT-B/16) w/o class token, w/ avg-pool @ 224x224
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'glsvit_base_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_huge_patch14_gap_224(pretrained: bool = False, **kwargs):
    """ ViT-Huge model (ViT-H/14) w/ no class token, avg pool
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'glsvit_huge_patch14_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_giant_patch16_gap_224(pretrained: bool = False, **kwargs):
    """ ViT-Giant (little-gg) model (ViT-g/16) w/ no class token, avg pool
    """
    model_args = dict(
        patch_size=16, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'glsvit_giant_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_clip_224(pretrained: bool = False, **kwargs):
    """ ViT-B/16 CLIP image tower
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'glsvit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_small_patch14_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-S/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'glsvit_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch14_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'glsvit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_large_patch14_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-L/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'glsvit_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_giant_patch14_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-G/14 for DINOv2
    """
    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5,
        mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, img_size=518, act_layer=nn.SiLU
    )
    model = _create_vision_transformer(
        'glsvit_giant_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_small_patch14_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-S/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'glsvit_small_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch14_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-B/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'glsvit_base_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_large_patch14_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-L/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'glsvit_large_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_giant_patch14_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-G/14 for DINOv2
    """
    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5, mlp_ratio=2.66667 * 2,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'glsvit_giant_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_siglip_224(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'glsvit_base_patch16_siglip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_siglip_256(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'glsvit_base_patch16_siglip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_siglip_512(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'glsvit_base_patch16_siglip_512', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_large_patch16_siglip_256(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'glsvit_large_patch16_siglip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_medium_patch16_reg4_256(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=True,
        no_embed_class=True, reg_tokens=4,
    )
    model = _create_vision_transformer(
        'glsvit_medium_patch16_reg4_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_medium_patch16_reg4_gap_256(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8,
        class_token=False, no_embed_class=True, reg_tokens=4, global_pool='avg',
    )
    model = _create_vision_transformer(
        'glsvit_medium_patch16_reg4_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsvit_base_patch16_reg8_gap_256(pretrained: bool = False, **kwargs):
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False,
        no_embed_class=True, global_pool='avg', reg_tokens=8,
    )
    model = _create_vision_transformer(
        'glsvit_base_patch16_reg8_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('glsdeit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('glsdeit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('glsdeit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit3_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_vision_transformer('glsdeit3_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit3_medium_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6)
    model = _create_vision_transformer('glsdeit3_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit3_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_vision_transformer('glsdeit3_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit3_large_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_vision_transformer('glsdeit3_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def glsdeit3_huge_patch14_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_vision_transformer('glsdeit3_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


if __name__ == "__main__":
    import timm
    import torch

    name = 'glsvit_base_patch16_224.orig_in21k'
    model = timm.create_model(name, num_classes=1000, pretrained=False)
    model.reset_classifier(num_classes=200)
    model.add_crop_cls()

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
