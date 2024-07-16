import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from .transformer import Transformer
from .download_load_pt_weights import load_pretrained_weights


class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViTGLSim(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.debugging = config.debugging

        self.seq_len_post_reducer = config.seq_len_post_reducer

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels, out_channels=config.hidden_size,
                kernel_size=(config.fh, config.fw), stride=(config.fh, config.fw))

        # Class token
        if 'cls' in config.classifier:
            self.class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Positional embedding
        if config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(
                config.seq_len_ind, config.hidden_size)

        # Transformer encoder
        self.encoder = Transformer(
            num_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd,
        )

        if config.encoder_norm:
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # reducer
        self.num_anchors_total = config.num_anchors_total

        if config.representation_size != config.hidden_size:
            reducer_resize = nn.Linear(config.hidden_size, config.representation_size)
        else:
            reducer_resize = nn.Identity()

        if config.reducer == 'cls':
            self.reducer = reducer_resize

        # aggregator
        if config.aggregator:
            self.aggregator = Transformer(
                num_layers=config.aggregator_num_hidden_layers,
                dim=config.representation_size,
                num_heads=config.representation_size // 64,
                ff_dim=config.representation_size * 4,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                layer_norm_eps=config.layer_norm_eps,
                sd=config.sd
                )

            if config.aggregator_norm:
                self.aggregator_norm = nn.LayerNorm(config.representation_size, eps=config.layer_norm_eps)

        # Classifier head
        if config.load_fc_layer:
            if config.classifier in ('cls', 'avg_cls', 'max_cls'):
                self.head = nn.Linear(config.representation_size, config.num_classes)
                if config.classifier in ('avg_cls', 'max_cls'):
                    reduce = 'mean' if config.classifier == 'avg_cls' else 'max'
                    self.head_reducer = nn.Sequential(
                        self.head,
                        Reduce('b s d -> b d', reduction=reduce)
                    )

        if config.anchor_size:
            self.anchor_size = config.anchor_size
            self.dynamic_top = config.dynamic_top
            self.sim_metric = config.sim_metric
            self.patch_equivalent = config.patch_equivalent
            self.anchor_random = config.anchor_random

            if config.anchor_class_token:
                self.anchor_class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Initialize weights
        self.init_weights()

        if pretrained:
            load_pretrained_weights(self, config, config.model_name)


    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        if hasattr(self, 'positional_embedding'):
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        if hasattr(self, 'class_token'):
            nn.init.constant_(self.class_token, 0)
        if hasattr(self, 'head'):
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

    @torch.no_grad()
    def get_crops(self, x, images):
        self.maybe_print('Obtaining crops: ', x.shape, images.shape)

        if hasattr(self, 'class_token'):
            g = x[:, :1, :]
            l = x[:, 1:, :]

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

        dist, ind = distances.topk(self.dynamic_top, dim=-1, largest=largest)
        self.maybe_print('Distances and 1-d indexes: ', distances.shape, dist.shape,
                         ind.shape, dist[0], ind[0])

        if self.anchor_random:
            ind = torch.randint(0, dist.shape[1], ind.shape, device=dist.get_device())

        ind_x = torch.div(ind, int(math.sqrt(l.shape[1])), rounding_mode='floor')
        ind_y = ind % int(math.sqrt(l.shape[1]))

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
            x_1 = min(max(x_f[i], x_i[i] + self.patch_equivalent), images.shape[-1])
            y_1 = min(max(y_f[i], y_i[i] + self.patch_equivalent), images.shape[-1])

            crop = images[i:i+1, :, x_0:x_1, y_0:y_1]
            crop = F.upsample_bilinear(crop, size=(images.shape[-1], images.shape[-1]))
            images_crops.append(crop)

        images_crops = torch.cat(images_crops, dim=0)
        return images_crops

    def patchify_tokenize(self, x):
        self.maybe_print('Before tokenizing: ', x.shape)

        x = self.patch_embedding(x)

        x = rearrange(x, 'b d gh gw -> b (gh gw) d')
        b, _, _ = x.shape

        if hasattr(self, 'class_token'):
            cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # b s+1 d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)

        self.maybe_print('After tokenizing: ', x.shape)
        return x

    def forward_encoder(self, x, vis=False):
        self.maybe_print('Before encoder: ', x.shape)

        x, inter, scores_soft, scores = self.encoder(x, vis=vis)
        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)

        self.maybe_print('After encoder: ', x.shape)
        return x, inter, scores_soft, scores

    def process_crops(self, x, x_crops):
        self.maybe_print('\nProcessing crops: ', x.shape, x_crops.shape)

        x_crops = self.patchify_tokenize(x_crops)

        x_crops, _, _, _ = self.forward_encoder(x_crops)

        x = torch.cat([x, x_crops], dim=1)

        self.maybe_print('After concatenating: ', x.shape)
        return x

    def forward_reducer(self, x):
        if hasattr(self, 'reducer'):
            if hasattr(self, 'class_token'):
                x = rearrange(x, 'b (k s) d -> b k s d', k=self.num_anchors_total)
                x = self.reducer(x[:, :, 0, :])
            else:
                x = self.reducer(x)

        self.maybe_print('\nAfter reducer: ', x.shape)
        return x

    def forward_aggregator(self, x):
        x = self.forward_reducer(x)

        if hasattr(self, 'aggregator'):
            x, _, _, _= self.aggregator(x)
            if hasattr(self, 'aggregator_norm'):
                x = self.aggregator_norm(x)

        self.maybe_print('After aggregator: ', x.shape)

        return x

    def classify(self, x):
        if hasattr(self, 'head') and hasattr(self, 'class_token'):
            if hasattr(self, 'head_reducer'):
                x = self.head_reducer(x)
            else:
                x = x[:, 0, :]
                x = self.head(x)

        self.maybe_print('After classifier head: ', x.shape)
        return x

    def forward(self, x, ret_dist=False):
        """
        x (tensor): b c h w -> b s d
        """
        if hasattr(self, 'anchor_size'):
            images = x

        x = self.patchify_tokenize(x)

        x, inter, scores_soft, scores = self.forward_encoder(x, vis=ret_dist)
        if ret_dist:
            x_norm = x

        if hasattr(self, 'anchor_size'):
            images_crops = self.get_crops(x, images)
            x = self.process_crops(x, images_crops)

        x = self.forward_aggregator(x)

        x = self.classify(x)

        if ret_dist:
            if hasattr(self, 'anchor_size'):
                return x, images_crops, x_norm, inter, scores_soft, scores
            else:
                return x, x_norm, inter, scores_soft, scores

        elif hasattr(self, 'anchor_size'):
            return x, images_crops
        return x

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)
