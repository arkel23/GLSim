import os
import math

import torch
from torchvision.utils import save_image

from glsim.data_utils.build_dataloaders import build_dataloaders
from glsim.other_utils.build_args import parse_inference_args
from glsim.train_utils.misc_utils import set_random_seed


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def adjust_args_general(args):
    if args.attention == 'mixer':
        args.model_name = args.model_name.replace('vit', 'mixer')
    if args.anchor_size:
        args.run_name = '{}_{}_{}_{}'.format(
            args.dataset_name, args.model_name, args.anchor_size, args.serial
        )
    else:
        args.run_name = '{}_{}_{}'.format(
            args.dataset_name, args.model_name, args.serial
        )

    args.results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def vis_dataset(args):
    if args.custom_mean_std:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    args = adjust_args_general(args)
    print(args)

    for split, loader in zip(['test', 'train'], [test_loader, train_loader]):
    # for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        for idx, (images, _) in enumerate(loader):
            images = inverse_normalize(images.data, mean, std)
            fp = os.path.join(args.results_dir, f'{split}_{idx}.png')
            save_image(images, fp, nrow=int(math.sqrt(images.shape[0])))

            if idx % args.log_freq == 0:
                print(f'{split} ({idx} / {len(loader)}): {fp}')

            if not args.vis_dataset_all:
                break

    return 0


def main():
    args = parse_inference_args()

    vis_dataset(args)

    return 0


if __name__ == '__main__':
    main()