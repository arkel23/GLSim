import math
import torch
import torch.utils.data as data
import torch.distributed as dist

from .datasets import get_set
from .build_transform import build_transform


def build_dataloaders(args):
    train_loader = get_loader(args, 'train')
    val_loader = get_loader(args, 'val')
    test_loader = get_loader(args, 'test')
    return train_loader, val_loader, test_loader


def get_loader(args, split):
    transform = build_transform(args=args, split=split)
    ds = get_set(args, split, transform)

    if split == 'train':
        shuffle = True
        drop_last = True
    elif args.shuffle_test:
        shuffle = True
        drop_last = False
    else:
        shuffle = False
        drop_last = False

    sampler = None
    if args.distributed:
        if split == 'train' and args.ra > 1:
            sampler = RASampler(
                ds, num_replicas=args.world_size, rank=args.rank, shuffle=True,
                num_repeats=args.ra
            )
        elif split == 'train':
            sampler = data.DistributedSampler(ds, num_replicas=args.world_size, rank=args.rank,
                                              shuffle=True, drop_last=True)
        else:
            sampler = data.SequentialSampler(ds)
        data_loader = data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.cpu_workers,
            pin_memory=args.pin_memory, sampler=sampler)
        return data_loader

    elif split == 'train' and args.ra > 1:
        shuffle = False
        sampler = RASampler(ds, num_replicas=1, rank=0, shuffle=True, num_repeats=args.ra)
        data_loader = data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.cpu_workers,
            pin_memory=args.pin_memory, sampler=sampler)
        return data_loader

    data_loader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.cpu_workers,
        drop_last=drop_last, pin_memory=args.pin_memory, sampler=sampler)
    return data_loader


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
