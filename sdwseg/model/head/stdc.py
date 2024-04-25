from typing import Union, List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.model import BUILD_NETWORK_REGISTRY

from .base_head import BaseHead
from ..utils import ConvBNReLU
from .fcn_head import FCNHead

__all__ = [
    'ContextPath'
]


class AttentionRefinementModule(nn.Module):
    """
    AttentionRefinementModule to apply on the last two backbone stages.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionRefinementModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_first = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), ConvBNReLU(out_channels, out_channels, kernel_size=1, bias=False, use_activation=False), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_first(x)
        y = self.attention_block(x)
        return torch.mul(x, y)


class FeatureFusionModule(nn.Module):
    """
    Fuse features from higher resolution aka, spatial feature map with features from lower resolution with high
     semantic information aka, context feature map.
    :param spatial_channels: num channels of input from spatial path.
    :param context_channels: num channels of input from context path.
    :param out_channels: num channels of feature fusion module.
    """

    def __init__(self, spatial_channels: int, context_channels: int, out_channels: int):
        super(FeatureFusionModule, self).__init__()
        self.spatial_channels = spatial_channels
        self.context_channels = context_channels
        self.out_channels = out_channels

        self.pw_conv = ConvBNReLU(spatial_channels + context_channels, out_channels, kernel_size=1, stride=1, bias=False)
        # TODO - used without bias in convolutions by mistake, try to reproduce with bias=True
        self.attention_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=1, use_normalization=False, bias=False),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feats, context_feats):
        feat = torch.cat([spatial_feats, context_feats], dim=1)
        feat = self.pw_conv(feat)
        atten = self.attention_block(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class ContextEmbedding(nn.Module):
    """
    ContextEmbedding module that use global average pooling to 1x1 to extract context information, and then upsample
    to original input size.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ContextEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.fixed_size = False

    def forward(self, x):
        out_height, out_width = x.size()[2:]
        x = self.context_embedding(x)
        return F.interpolate(x, size=(out_height, out_width), mode="nearest")

    def to_fixed_size(self, upsample_size: Union[list, tuple]):
        if self.fixed_size:
            return
        self.fixed_size = True

        self.context_embedding.add_module("upsample", nn.Upsample(scale_factor=upsample_size, mode="nearest"))

        self.forward = self.context_embedding.forward


@BUILD_NETWORK_REGISTRY.register()
class ContextPath(BaseHead):
    """
    ContextPath in backbone output both the Spatial path and Context path

    :param in_channels: output channels of Backbone
    :param context_fuse_channels: num channels of the fused context path.
    :param ffm_channels: num of output channels of Feature Fusion Module.
    :param num_classes:
    :param dropout_ratio:
    """

    def __init__(self, in_channels: List[int],
                 num_classes: int,
                 context_fuse_channels: int = 128,
                 ffm_channels: int = 256,
                 loss_cfg: Optional[Dict] = None,
                 dropout_ratio: float = 0.2,
                 ignore_index: Optional[int] = None):
        super(ContextPath, self).__init__(num_classes, loss_cfg, ignore_index)

        self.context_fuse_channels = context_fuse_channels

        # context feature map
        self.context_embedding = ContextEmbedding(in_channels[-1], context_fuse_channels)

        # the spatial feature map
        self.arm = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.arm.append(AttentionRefinementModule(in_channels[-(i + 1)], context_fuse_channels))
            self.upsample.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBNReLU(context_fuse_channels, context_fuse_channels, kernel_size=3, padding=1, stride=1, bias=False)
            ))
        self.ffm = FeatureFusionModule(spatial_channels=in_channels[0], context_channels=context_fuse_channels, out_channels=ffm_channels)
        self.head = FCNHead(ffm_channels, self.num_classes,
                            ffm_channels, 2, dropout_ratio=dropout_ratio,
                            loss_cfg=None, concat_input=False, norm_cfg=dict(type='BatchNorm2d'))
        return

    def forward(self, x_stagegs, shape):

        ce_feats = self.context_embedding(x_stagegs[-1])

        for i in range(len(self.arm)):
            feat_arm = self.arm[i](x_stagegs[-(i + 1)]) + ce_feats
            ce_feats = self.upsample[i](feat_arm)

        out = self.ffm(spatial_feats=x_stagegs[0], context_feats=ce_feats)

        return self.head([out], shape)

    def prep_for_conversion(self, input_size):
        if input_size[-2] % 32 != 0 or input_size[-1] % 32 != 0:
            raise ValueError(f"Expected image dimensions to be divisible by 32, got {input_size[-2]}x{input_size[-1]}")

        context_embedding_up_size = (input_size[-2] // 32, input_size[-1] // 32)
        self.context_embedding.to_fixed_size(context_embedding_up_size)
