import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from engine.model import BUILD_NETWORK_REGISTRY

from ..utils import ConvModule
from .base_head import BaseHead

__all__ = [
    'FCNHead',
]


@BUILD_NETWORK_REGISTRY.register()
class FCNHead(BaseHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 channels: int,
                 num_convs=2,
                 dropout_ratio: float = 0.1,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 align_corners=False,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 loss_cfg: Optional[Dict] = None,
                 in_index: int = -1,
                 ignore_index: Optional[int] = None):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(num_classes, loss_cfg, ignore_index)
        if num_convs == 0:
            assert in_channels == channels

        conv_padding = (kernel_size // 2) * dilation
        convs = list()
        convs.append(
            ConvModule(
                in_channels,
                channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                in_channels + channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.conv_seg = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.in_index = in_index
        self.align_corners = align_corners
        return

    def forward(self, x_stages, shape):
        feat = self.convs(x_stages[self.in_index])
        if self.concat_input:
            feat = self.conv_cat(torch.cat([x_stages[self.in_index], feat], dim=1))

        if self.dropout is not None:
            feat = self.dropout(feat)
        out = self.conv_seg(feat)
        out = F.interpolate(out, shape, mode='bilinear', align_corners=self.align_corners)
        return out

