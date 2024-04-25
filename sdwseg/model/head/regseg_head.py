from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.model import BUILD_NETWORK_REGISTRY

from .base_head import BaseHead
from ..utils import ConvBNReLU

__all__ = [
    'RegSegDecoder',
]


@BUILD_NETWORK_REGISTRY.register()
class RegSegDecoder(BaseHead):
    """
    This implementation follows the paper. No 'pattern' in this decoder, so it is specific to 3 stages
    """

    def __init__(self, in_channels: List[int],
                 projection_out_channels: List[int],
                 num_classes: int,
                 head_channels: int,
                 interpolation: str = 'bilinear',
                 dropout: float = 0.0,
                 align_corners: bool = False,
                 upsample_factor: int = 4,
                 loss_cfg: Optional[Dict] = None,
                 ignore_index: Optional[int] = None
                 ):
        super(RegSegDecoder, self).__init__(num_classes, loss_cfg, ignore_index)

        assert len(in_channels) == len(projection_out_channels) == 3, "This decoder is specific for 3 stages"

        self.projections = nn.ModuleList(
            [ConvBNReLU(in_channel, out_channel, 1, bias=False) for in_channel, out_channel in zip(in_channels, projection_out_channels)]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation, align_corners=True)
        mid_channels = projection_out_channels[1]
        self.conv_bn_relu = ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels // 2, kernel_size=3, padding=1, bias=False)

        out_channels = mid_channels // 2 + projection_out_channels[0]  # original implementation: concat
        head = list()
        head.append(ConvBNReLU(out_channels, head_channels, 3, bias=False, padding=1))
        if dropout > 0:
            head.append(nn.Dropout(dropout, inplace=False))
        head.append(nn.Conv2d(head_channels, self.num_classes, 1))
        head.append(nn.Upsample(scale_factor=upsample_factor, mode=interpolation, align_corners=align_corners))
        self.head = nn.Sequential(*head)
        return

    def forward(self, x_stages, shape):
        proj2 = self.projections[2](x_stages[2])
        proj2 = self.upsample(proj2)
        proj1 = self.projections[1](x_stages[1])
        proj1 = proj1 + proj2
        proj1 = self.conv_bn_relu(proj1)
        proj1 = self.upsample(proj1)
        proj0 = self.projections[0](x_stages[0])
        proj0 = torch.cat((proj1, proj0), dim=1)
        out = self.head(proj0)
        if out.shape[2:] != shape:
            out = F.interpolate(out, shape, mode='bilinear', align_corners=self.align_corners)
        return out


