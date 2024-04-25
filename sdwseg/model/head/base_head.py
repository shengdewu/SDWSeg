import abc
import torch.nn as nn
from typing import Optional, Dict, Tuple

from engine.loss import LossKeyCompose

from ..utils.accuracy import accuracy

__all__ = [
    'BaseHead',
]


class BaseHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 loss_cfg: Optional[Dict] = None,
                 ignore_index: Optional[int] = None):
        super(BaseHead, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.with_loss = False
        if loss_cfg is not None:
            self.with_loss = True
            self.loss_fn = LossKeyCompose(dict(regseg=loss_cfg))
        return

    @abc.abstractmethod
    def forward(self, x_stages, shape):
        raise NotImplemented('the forward of BaseHead')

    def forward_train(self, x_stages, shape, gt_semantic_seg):
        x = self(x_stages, shape)
        loss = self.loss_fn(dict(regseg=[(x, gt_semantic_seg)]))
        acc = accuracy(x.detach(), gt_semantic_seg, ignore_index=self.ignore_index)
        if isinstance(acc, Tuple):
            acc = acc[0]
        return loss, acc.item()

    def preparate_deploy(self):
        if self.with_loss:
            delattr(self, 'loss_fn')
            self.with_loss = False
        return
