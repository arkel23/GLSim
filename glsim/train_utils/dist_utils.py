# https://github.com/rwightman/pytorch-image-models/blob/02daf2ab943ce2c1646c4af65026114facf4eb22/timm/utils/distributed.py
# https://github.com/rwightman/pytorch-image-models/blob/02daf2ab943ce2c1646c4af65026114facf4eb22/timm/utils/model.py#L15
import torch
import torch.distributed as dist


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)
