import random

import numpy as np
import torch
import wandb


def count_params_module_list(module_list):
    return sum([count_params_single(model) for model in module_list])


def count_params_single(model):
    return sum([p.numel() for p in model.parameters()])


def count_params_trainable(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def set_random_seed(seed=0, numpy=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if numpy:
        np.random.seed(seed)
    return 0


def summary_stats(epochs, time_total, best_acc, best_epoch, max_memory,
                  no_params, class_deviation, debugging=False):
    time_avg = round((time_total / epochs) / 60, 2)
    best_time = round((time_avg * best_epoch) / 60, 2)
    time_total = round(time_total / 60, 2)  # mins
    no_params = round(no_params / (1e6), 2)  # millions of parameters
    max_memory = round(max_memory, 2)

    print('''Total run time (minutes): {}
          Average time per epoch (minutes): {}
          Best accuracy (%): {} at epoch {}. Time to reach this (minutes): {}
          Class deviation: {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          '''.format(time_total, time_avg, best_acc, best_epoch, best_time,
                     class_deviation, max_memory, no_params))

    if not debugging:
        wandb.run.summary['time_total'] = time_total
        wandb.run.summary['time_avg'] = time_avg
        wandb.run.summary['best_acc'] = best_acc
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_time'] = best_time
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['class_deviation'] = class_deviation
    return 0


def stats_test(test_acc, class_deviation, max_memory, no_params,
               time_total, num_images, debugging=False):

    throughput = round(num_images / time_total, 2)
    no_params = round(no_params / (1e6), 2)  # millions of parameters
    max_memory = round(max_memory, 2)

    print('''Throughput (images / s): {}
          Test accuracy (%): {}
          Class deviation (%): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          '''.format(throughput, test_acc, class_deviation, max_memory, no_params))

    if not debugging:
        wandb.run.summary['test_acc'] = test_acc
        wandb.run.summary['class_deviation'] = class_deviation
        wandb.run.summary['throughput'] = throughput
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
    return 0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def inverse_normalize(tensor, norm_custom=False):
    if norm_custom:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor
