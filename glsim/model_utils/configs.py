import math


REDUCERS = (None, False, True, 'linear', 'pool', 'cls', 'attention_pool', 'std')
HEADS = (None, False, 'cls', 'pool', 'seq', 'seq_cls', 'avg_cls', 'max_cls',
         'first', 'avg_first', 'max_first',
         'attention_pool', 'attention_pool_cls')
AUX = (None, False, 'cls', 'pool', 'multi_cont', 'shared', 'consistency')


class ViTConfig():
    def __init__(self,
                 model_name: str = 'vit_b16',
                 debugging: bool = None,
                 test_flip: bool = None,
                 image_size: int = None,
                 patch_size: tuple() = None,
                 load_fc_layer: bool = None,
                 classifier: str = None,
                 num_classes: int = None,
                 num_channels: int = None,
                 shared_patchifier: bool = None,

                 classifier_aux: str = None,
                 head_norm: bool = None,
                 head_act: bool = None,
                 head_dropout_all: bool = None,
                 head_dropout_prob: float = None,

                 anchor_size=None,
                 num_anchors=None,
                 dynamic_anchor: bool = None,
                 dynamic_top: int = None,
                 anchor_pool_avg: bool = None,
                 anchor_pool_stride: int = None,
                 sim_metric: str = None,
                 anchor_random: bool = None,
                 anchor_class_token: bool = None,
                 anchor_conv_stem: bool = None,
                 anchor_conv_stride: int = None,
                 anchor_resize_size: bool = None,

                 pos_embedding_type: str = None,
                 hidden_size: int = None,
                 intermediate_size: int = None,
                 num_attention_heads: int = None,
                 num_hidden_layers: int = None,
                 encoder_norm: bool = None,

                 reducer: str = None,

                 aggregator: bool = None,
                 aggregator_num_hidden_layers: int = None,
                 aggregator_norm: bool = None,

                 representation_size: int = None,
                 attention_probs_dropout_prob: float = None,
                 hidden_dropout_prob: float = None,
                 sd: float = None,
                 layer_norm_eps: float = None,
                 hidden_act: str = None,
                 url: str = None,
                 print_attr: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        default = CONFIGS[model_name]

        input_args = locals()
        for k, v in input_args.items():
            if k in default['config'].keys():
                setattr(self, k, v if v is not None else default['config'][k])

        self.assertions_corrections()
        self.calc_dims()
        if print_attr:
            print(vars(self))

    def as_tuple(self, x):
        return x if isinstance(x, tuple) else (x, x)

    def calc_dims(self):
        h, w = self.as_tuple(self.image_size)  # image sizes
        self.fh, self.fw = self.as_tuple(self.patch_size)  # patch sizes
        self.gh, self.gw = h // self.fh, w // self.fw  # number of patches
        self.seq_len_ind = self.gh * self.gw  # sequence length individual image
        assert(self.gh ** 2 == self.seq_len_ind), 'Only accept same H=W features'

        if 'cls' in self.classifier:
            self.seq_len_ind += 1

        self.patch_equivalent = self.image_size // self.gh

        if self.dynamic_anchor:
            self.anchor_size = self.patch_equivalent
            self.num_anchors = 1

        if self.anchor_size:
            if isinstance(self.num_anchors, list):
                for anchor in self.anchor_size:
                    assert (anchor % self.patch_equivalent) == 0, \
                        f'anchor size {anchor} must be multiple of patch_equivalent {self.patch_equivalent}'
                self.num_anchors_total = sum(self.num_anchors) + 1
            else:
                assert (self.anchor_size % self.patch_equivalent) == 0, \
                    f'anchor size {self.anchor_size} must be multiple of patch_equivalent {self.patch_equivalent}'
                self.num_anchors_total = self.num_anchors + 1
        else:
            self.num_anchors_total = 1

        if self.anchor_resize_size and (self.anchor_resize_size != self.image_size):
            anchor_image_size = self.anchor_resize_size
        else:
            anchor_image_size = self.image_size

        if self.anchor_conv_stride:
            assert anchor_image_size, 'Needs to input --anchor_resize_size when using anchor_conv_stride'
            self.seq_len_anchor_ind = (((anchor_image_size - self.fh) // 
                                        self.anchor_conv_stride) + 1) ** 2
        elif anchor_image_size != self.image_size:
            self.seq_len_anchor_ind = (anchor_image_size // self.fh) ** 2
        else:
            self.seq_len_anchor_ind = self.seq_len_ind

        if 'cls' in self.classifier and \
            ((self.anchor_conv_stride) or (anchor_image_size != self.image_size)):
            self.seq_len_anchor_ind += 1

        if self.anchor_size:
            self.seq_len_total = self.seq_len_ind + (self.seq_len_anchor_ind * self.num_anchors)
        else:
            self.seq_len_total = self.seq_len_ind

        if self.reducer in ['cls']:
            self.seq_len_post_reducer = self.num_anchors_total
        else:
            self.seq_len_post_reducer = self.seq_len_total

        if not self.reducer:
            self.representation_size = self.hidden_size

    def assertions_corrections(self):
        assert self.classifier in HEADS, f'Choose from {HEADS}'
        assert self.classifier_aux in AUX, f'Choose from {AUX}'

        if self.reducer == 'cls':
            assert 'cls' in self.classifier

        if self.classifier_aux == 'cls':
            assert 'cls' in self.classifier

        if not self.sim_metric:
            self.sim_metric = 'cos'


    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return str(vars(self))


class ResnetConfig(ViTConfig):
    def __init__(self, model_name,
                 **kwargs):
        super().__init__(model_name='vit_b16',
                         classifier='pool', **kwargs)
        self.model_name = model_name
        self.adjust_resnet()

    def adjust_resnet(self):
        if self.reducer == 'cls':
            self.reducer = 'pool'

        self.encoder_norm = False

        self.patch_size = (32, 32)
        self.fh, self.fw = self.as_tuple(self.patch_size)
        self.gh, self.gw = self.image_size // self.fh, self.image_size // self.fw  # number of patches
        self.seq_len_ind = self.gh * self.gw  # sequence length individual image

        self.hidden_size = 2048
        self.representation_size = 512
        if not self.reducer:
            self.representation_size = self.hidden_size
        self.intermediate_size = self.representation_size * 4
        self.num_attention_heads = self.representation_size // 64

        self.patch_equivalent = self.image_size // self.gh
        self.seq_len_total = self.seq_len_ind
        self.seq_len_post_reducer = self.seq_len_total


def get_base_config():
    """Base ViT config ViT"""
    return dict(
        debugging=False,
        test_flip=False,
        image_size=224,
        patch_size=(16, 16),
        load_fc_layer=True,
        classifier='cls',
        num_classes=1000,
        num_channels=3,
        shared_patchifier=False,

        classifier_aux=None,
        head_norm=False,
        head_act=False,
        head_dropout_all=False,
        head_dropout_prob=0,

        se=None,

        num_anchors=None,
        anchor_size=None,
        dynamic_anchor=False,
        dynamic_top=2,
        sim_metric=None,
        anchor_random=False,
        anchor_class_token=False,
        anchor_conv_stem=False,
        anchor_conv_stride=None,
        anchor_resize_size=False,
        exclude_bottom_k=None,

        attention='vanilla',
        pos_embedding_type='learned',
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        encoder_norm=True,

        reducer=None,

        aggregator=False,
        aggregator_num_hidden_layers=1,
        aggregator_norm=False,

        representation_size=768,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
        sd=0.0,
        layer_norm_eps=1e-12,
        hidden_act='gelu',
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(
        patch_size=(32, 32),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz'))
    return config


def get_b8_config():
    """Returns the ViT-B/8 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(8, 8),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_8.npz'))
    return config


def get_s16_config():
    """Returns the ViT-S/16 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        representation_size=384,
    ))
    return config


def get_s32_config():
    """Returns the ViT-S/32 configuration."""
    config = get_s16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_s8_config():
    """Returns the ViT-S/8 configuration."""
    config = get_s16_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_t16_config():
    """Returns the ViT-T configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=192,
        intermediate_size=768,
        num_attention_heads=3,
        representation_size=192,
    ))
    return config


def get_t32_config():
    """Returns the ViT-T/32 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_t8_config():
    """Returns the ViT-T/8 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_t4_config():
    """Returns the ViT-T/4 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(4, 4)))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        representation_size=1024,
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz'
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(
        patch_size=(32, 32),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz'))
    return config


def get_h14_config():
    """Returns the ViT-H/14 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=1280,
        intermediate_size=5120,
        num_attention_heads=16,
        num_hidden_layers=32,
        representation_size=1280,
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz'
    ))
    config.update(dict(patch_size=(14, 14)))
    return config


CONFIGS = {
    'vit_t4': {
        'config': get_t4_config(),
    },
    'vit_t8': {
        'config': get_t8_config(),
    },
    'vit_t16': {
        'config': get_t16_config(),
    },
    'vit_t32': {
        'config': get_t32_config(),
    },
    'vit_s8': {
        'config': get_s8_config(),
    },
    'vit_s16': {
        'config': get_s16_config(),
    },
    'vit_s32': {
        'config': get_s32_config(),
    },
    'vit_b8': {
        'config': get_b8_config(),
    },
    'vit_b16': {
        'config': get_b16_config(),
    },
    'vit_b32': {
        'config': get_b32_config(),
    },
    'vit_l16': {
        'config': get_l16_config(),
    },
    'vit_l32': {
        'config': get_l32_config(),
    },
    'vit_h14': {
        'config': get_h14_config(),
    }
}
