import torch
import torch.nn.functional as F


def cont_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack(
        [labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + \
        (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


def multi_cont_loss(features, labels=None, norm_ind=False, temperature=0.07, base_temperature=0.07):
    # Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    # Contrastive Deep Supervision also used same SupConLoss
    # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/ImageNet/Contrastive_Deep_Supervision/loss.py
    device = features.device

    # normalize each group of features individually as in CDS
    # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/ImageNet/Contrastive_Deep_Supervision/resnet.py
    if norm_ind:
        features = torch.split(features, 1, dim=1)
        features = [F.normalize(ft, dim=2) for ft in features]
        features = torch.cat(features, dim=1)
    else:
        features = F.normalize(features, dim=2)

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]

    if labels is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    else:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss
