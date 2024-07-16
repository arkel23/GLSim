import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    https://github.com/jimitshah77/plant-pathology/blob/master/bilinear-efficientnet-focal-loss-label-smoothing.ipynb
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, gamma=2.0, alpha=None, smoothing=False, ls=0.1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha,list):
            # weight for the classes (either do not use or use class distribution)
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ls = ls if smoothing else 0

    def forward(self, input, target):
        logprobs = F.log_softmax(input, dim=-1)

        nll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        if self.alpha is not None:
            if self.alpha.type() != input.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            nll_loss = nll_loss * at

        if self.ls > 0:
            nll_loss = (1 - self.ls) * nll_loss + self.ls * (logprobs.mean(dim=-1))

        pt = nll_loss.data.exp()

        loss = - ((1 - pt) ** self.gamma) * nll_loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
