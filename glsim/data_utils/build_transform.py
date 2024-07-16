from torchvision import transforms
from timm.data import create_transform

from .augmentations import CIFAR10Policy, SVHNPolicy, ImageNetPolicy


MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'tinyin': (0.4802, 0.4481, 0.3975),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5070, 0.4865, 0.4409),
    'svhn': (0.4377, 0.4438, 0.4728),
    'cub': (0.3659524, 0.42010019, 0.41562049)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'tinyin': (0.2770, 0.2691, 0.2821),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'svhn': (0.1980, 0.2010, 0.1970),
    'cub': (0.07625843, 0.04599726, 0.06182727)
}


def build_deit_transform(args, is_train):
    ''' taken from DeiT paper
    https://arxiv.org/abs/2012.12877
    https://github.com/facebookresearch/deit/blob/main/main.py'''
    image_size = args.image_size

    # augmentation and random erase params
    args.jitter = 0.4
    args.aa = 'rand-m9-mstd0.5-inc1'
    args.smoothing = 0.1
    args.train_interpolation = 'bicubic'
    args.repeated_aug = True
    args.reprob = 0.25
    args.remode = 'pixel'
    args.recount = 1
    args.resplit = False

    mean = MEANS['imagenet']
    std = STDS['imagenet']
    if args.custom_mean_std:
        mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
        std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

    resize_im = image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            jitter=args.jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                image_size, padding=4)
        return transform

    t = []
    if resize_im:
        # to maintain same ratio w.r.t. 224 images
        size = int((256 / 224) * image_size)
        t.append(transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean,
                                  std=std))
    return transforms.Compose(t)


def standard_transform(args, is_train):
    image_size = args.image_size
    resize_size = args.resize_size
    test_resize_size = args.test_resize_size

    mean = MEANS['imagenet']
    std = STDS['imagenet']
    if args.custom_mean_std:
        mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
        std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

    if 'cifar' in args.dataset_name and image_size == 32:
        aa = CIFAR10Policy()
    elif args.dataset_name == 'svhn' and image_size == 32:
        aa = SVHNPolicy()
    else:
        aa = ImageNetPolicy()

    t = []

    if is_train:

        if args.affine:
            t.append(transforms.Resize(
                (resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.RandomAffine(degrees=15, scale=(0.85, 1.15),
                                             interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.RandomCrop((image_size, image_size)))
        elif args.random_resized_crop:
            t.append(transforms.RandomResizedCrop(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC))
        elif args.square_resize_random_crop:
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.RandomCrop(image_size))
        elif args.short_side_resize_random_crop:
            t.append(transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.RandomCrop((image_size, image_size)))
        elif args.square_center_crop:
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(image_size))

        if args.horizontal_flip:
            t.append(transforms.RandomHorizontalFlip())
        if args.vertical_flip:
            t.append(transforms.RandomVerticalFlip())
        if args.jitter_prob > 0:
            t.append(transforms.RandomApply([transforms.ColorJitter(
                brightness=args.jitter_bcs, contrast=args.jitter_bcs,
                saturation=args.jitter_bcs, hue=args.jitter_hue)], p=args.jitter_prob))
        if args.greyscale > 0:
            t.append(transforms.RandomGrayscale(p=args.greyscale))
        if args.blur > 0:
            t.append(transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=args.blur))
        if args.solarize_prob > 0:
            t.append(transforms.RandomApply(
                [transforms.RandomSolarize(args.solarize, p=args.solarize_prob)]))
        if args.aa:
            t.append(aa)
        if args.randaug:
            t.append(transforms.RandAugment())
        if args.trivial_aug:
            t.append(transforms.TrivialAugmentWide())
    else:
        if ((args.dataset_name in ['cifar10', 'cifar100', 'svhn'] and image_size == 32)
           or (args.dataset_name == 'tinyin' and image_size == 64)):
            t.append(transforms.Resize(image_size))
        else:
            if args.test_resize_directly:
                t.append(transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC))
            else:
                t.append(transforms.Resize(
                    (test_resize_size, test_resize_size),
                    interpolation=transforms.InterpolationMode.BICUBIC))
                t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    if is_train and args.re > 0:
        t.append(transforms.RandomErasing(
            p=args.re, scale=(0.02, args.re_sh), ratio=(args.re_r1, 3.3)))
    transform = transforms.Compose(t)
    print(transform)
    return transform


def build_transform(args, split):
    is_train = True if split == 'train' else False

    if args.deit_recipe:
        transform = build_deit_transform(args, is_train)
    else:
        transform = standard_transform(args, is_train)

    return transform
