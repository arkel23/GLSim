import math

import cv2
import numpy as np
from PIL import Image
from einops import rearrange, reduce

import torch
from torch.nn import functional as F

from glsim.train_utils.misc_utils import inverse_normalize


def save_pil_images(images, fp):
    images.save(fp)
    print(f'Saved file to : {fp}')
    return 0


def apply_mask(img, mask, top_k=8, th=False, pow=True, color=True):
    '''
    img are pytorch tensors of size C x S1 x S2, range 0-255 (unnormalized)
    mask are pytorch tensors of size (S1 / P * S2 / P) of floating values (range prob -1 ~ 1)
    heatmap combination requires using opencv (and therefore numpy arrays)
    '''

    if th:
        # compute threshold for binarizing
        highest, _ = mask.topk(top_k, dim=-1, largest=True)
        th_value = highest[-1].item()
        mask = torch.where(mask >= th_value, 1, 0)
    elif pow:
        mask = (mask ** 4)

    # 1d sequence to 2d feature map and convert to numpy array
    h = int(math.sqrt(mask.shape[0]))
    mask = rearrange(mask, '(h w) -> h w', h=h)
    mask = mask.cpu().numpy()

    # print('Mean and std of mask: ', np.mean(mask), np.std(mask))

    if color:
        mask = cv2.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        mask = mask.astype('uint8')

    mask = cv2.resize(mask, (img.shape[-1], img.shape[-1]))

    img = rearrange(img.cpu().numpy(), '1 c h w -> h w c')
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')

    if color:
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    else:
        mask = rearrange(mask, 'h w -> h w 1')        

    if color:
        result = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
    else:
        result = (mask * img)
        result = cv2.normalize(
            result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = result.astype('uint8')

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(result)

    return result


def attention_rollout(scores_soft):
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = torch.stack(scores_soft).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1), device=att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size(), device=att_mat.device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    mask = joint_attentions[-1][0, 1:]
    # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # mask = v[0, 1:].reshape(grid_size, grid_size)

    return mask


def calc_attention(attention):
    attention = torch.mean(attention[0].squeeze(), dim=1)[0, 1:]
    return attention


def calc_distance_glsim(tokens):
    g = tokens[0, :1, :]
    l = tokens[0, 1:, :]
    distances = F.cosine_similarity(g, l, dim=-1)

    return distances


def calc_distance_psm(scores_soft, head=0):
    att_mat = torch.stack(scores_soft).squeeze(1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(att_mat.size(), device=att_mat.device)
    joint_attentions[0] = att_mat[0]

    for n in range(1, att_mat.size(0)):
        joint_attentions[n] = torch.matmul(att_mat[n], joint_attentions[n-1])

    # Attention from the output CLS token to the input space.
    mask = joint_attentions[-1][head, 0, 1:]
    for n in range(1, joint_attentions[-1].shape[0]):
        mask += joint_attentions[-1][n, 0, 1:]
    return mask


def calc_distance_maws(scores_soft, scores):
    a = reduce(scores_soft.squeeze(), 'h s1 s2 -> s1 s2', 'mean')[0, 1:]
    b = F.softmax(scores.squeeze(), dim=-2)[:, 1:, 0]
    b = reduce(b, 'h s1 -> s1', 'mean')
    ma = a * b
    return ma


def calc_asim(x_norm, scores_soft):
    distances = calc_distance_glsim(x_norm)

    attn = attention_rollout(scores_soft)

    asim = ((distances - torch.min(distances)) / torch.max(distances)) * ((attn - torch.min(attn)) / torch.max(attn))

    return asim


def make_heatmaps(args, fp, img, inter, scores_soft, scores, x_norm,
                  vis_mask_all=False, save=True):

    if vis_mask_all:

        for mechanism in args.vis_mask_list:
            args.vis_mask = mechanism
            make_heatmaps(args, fp, img, inter, scores_soft, scores, x_norm)

        return 0

    img = inverse_normalize(img.detach().clone(), args.custom_mean_std)

    if args.vis_mask.split('_')[-1].isnumeric():
        select = int(args.vis_mask.split('_')[-1])
    elif len(args.vis_mask.split('_')) == 1:
        if 'maws' in args.vis_mask:
            select = 10
        elif 'psm' in args.vis_mask:
            select = 0
        else:
            select = 11

    if args.vis_mask == 'asim':
        mask = calc_asim(x_norm,  scores_soft)
    elif args.vis_mask == 'glsim_norm':
        mask = calc_distance_glsim(x_norm)
    elif 'glsim' in args.vis_mask:
        mask = calc_distance_glsim(inter[select])
    elif 'attention' in args.vis_mask:
        mask = calc_attention(scores_soft[select])        
    elif 'rollout' in args.vis_mask:
        mask = attention_rollout(scores_soft[:select])
    elif 'psm' in args.vis_mask:
        mask = calc_distance_psm(scores_soft, select)
    elif 'maws' in args.vis_mask:
        mask = calc_distance_maws(scores_soft[select], scores[select])

    img_masked = apply_mask(img, mask, top_k=args.dynamic_top, th=args.vis_th_topk,
                            pow=args.vis_mask_pow, color=args.vis_mask_color)

    if not save:
        return img_masked

    fp = fp.replace('.png', f'_{args.vis_mask}.png')

    save_pil_images(img_masked, fp)

    return img_masked
