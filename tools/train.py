import os
import time
import random

import wandb
from timm.optim import create_optimizer
from timm.loss import LabelSmoothingCrossEntropy
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from glsim.data_utils.build_dataloaders import build_dataloaders
from glsim.model_utils.build_model import build_model
from glsim.other_utils.build_args import parse_train_args
from glsim.train_utils.misc_utils import summary_stats, stats_test, set_random_seed
from glsim.train_utils.scheduler import build_scheduler
from glsim.train_utils.trainer import Trainer
from glsim.train_utils.focal_loss import FocalLoss


IGNORE = ('project_name', 'ckpt_path', 'transfer_learning', 'test_only', 'batch_size',
          'vis_errors', 'vis_errors_save', 'distributed', 'epochs', 'debugging', 'test_multiple')


def adjust_args_general(args):
    if args.anchor_size:
        args.run_name = '{}_{}_{}_{}'.format(
            args.dataset_name, args.model_name, args.anchor_size, args.serial
        )
    else:
        args.run_name = '{}_{}_{}'.format(
            args.dataset_name, args.model_name, args.serial
        )

    args.results_dir = os.path.join(args.results_dir, args.run_name)


def build_environment(args):
    if args.ckpt_path:
        args_temp = vars(torch.load(args.ckpt_path, map_location=torch.device('cpu'))['config'])
        for k, v in args_temp.items():
            if k not in IGNORE:
                if ((k == 'anchor_size' and getattr(args, k, None) is not None) or
                    (k == 'num_anchors' and getattr(args, k, None) is not None) or
                    (k == 'sim_metric' and getattr(args, k, None) is not None) or
                    (k == 'dynamic_top' and getattr(args, k, None) is not None) or
                    (k == 'exclude_bottom_k' and getattr(args, k, None) is not None) or
                    (k == 'df_classid_classname' and getattr(args, k, None) is not None) or
                    (k == 'dataset_root_path' and getattr(args, k, None) is not None) or
                    (k == 'test_resize_size' and (
                        getattr(args, k) >= args_temp['image_size'] and
                        getattr(args, k) < v))):
                    pass
                else:
                    setattr(args, k, v)

    if args.serial is None:
        args.serial = random.randint(0, 1000)
    # Set device and random seed
    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # model and criterion
    model = build_model(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.zero_grad()

    if args.focal_gamma:
        criterion = FocalLoss(args.focal_gamma, smoothing=args.ls)
    elif args.ls:
        criterion = LabelSmoothingCrossEntropy(args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # loss and optimizer
    optimizer = create_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, train_loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if not args.ckpt_path:
        adjust_args_general(args)
    os.makedirs(args.results_dir, exist_ok=True)

    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader


def main():
    time_start = time.time()

    args = parse_train_args()

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader = build_environment(args)

    trainer = Trainer(args, model, criterion, optimizer, lr_scheduler,
                      train_loader, val_loader, test_loader)

    if args.test_only:
        if not args.vis_errors and not args.debugging and not args.test_offline:
            wandb.init(config=args, project=args.project_name, entity=args.entity)
            wandb.run.name = args.run_name
        time_start = time.time()
        print(args, model.cfg)

        test_acc, max_memory, no_params, class_deviation = trainer.test()

        if not args.debugging:
            time_total = time.time() - time_start

            if args.test_multiple:
                num_images = (args.test_multiple + 1) * args.num_images_test
            else:
                num_images = args.num_images_test

            stats_test(test_acc, class_deviation, max_memory, no_params,
                       time_total, num_images, (args.vis_errors or args.test_offline))
            if not args.vis_errors and not args.test_offline:
                wandb.finish()
    else:
        if args.local_rank == 0:
            if not args.debugging:
                wandb.init(config=args, project=args.project_name, entity=args.entity)
                wandb.run.name = args.run_name
            if not args.distributed:
                print(model, model.cfg)
            print(args)

        best_acc, best_epoch, max_memory, no_params, class_deviation = trainer.train()

        # summary stats
        if args.local_rank == 0 and not args.debugging:
            time_total = time.time() - time_start
            summary_stats(args.epochs, time_total, best_acc, best_epoch, max_memory,
                          no_params, class_deviation, args.debugging)
            wandb.finish()


if __name__ == '__main__':
    main()
