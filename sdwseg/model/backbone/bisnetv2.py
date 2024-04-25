from typing import Tuple, Optional

import torch
from torch.functional import F
import torch.nn as nn

from engine.model import BUILD_NETWORK_REGISTRY

from ..utils import ConvBNReLU, ConvModule

__all__ = [
    'BiSeNetV2'
]


class StemBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)

        self.left = nn.Sequential(
            ConvBNReLU(out_dim, out_dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            ConvBNReLU(out_dim // 2, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.right = nn.MaxPool2d(3, stride=2, padding=1)

        self.fuse = ConvBNReLU(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

        return

    def forward(self, x):
        s = self.conv(x)
        left = self.left(s)
        right = self.right(s)
        concat = torch.concat([left, right], dim=1)
        return self.fuse(concat)


class ContextEmbeddingBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ContextEmbeddingBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_dim)

        self.conv_1x1 = ConvModule(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=dict(type='BatchNorm2d'), act_cfg=dict(type='ReLU'))
        self.conv_3x3 = ConvModule(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, norm_cfg=dict(type='BatchNorm2d'), act_cfg=dict(type='ReLU'))

        return

    def forward(self, x):
        bn = self.bn(x.mean((2, 3), keepdim=True))
        return self.conv_3x3(x + self.conv_1x1(bn))


class GatherAndExpansionLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, expand, stride=1):
        super(GatherAndExpansionLayer, self).__init__()
        mid_channel = in_dim * expand

        self.conv1 = ConvBNReLU(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        if stride == 1:
            self.dwconv = ConvBNReLU(in_dim, mid_channel, kernel_size=3, stride=stride, padding=1, groups=in_dim, bias=False)
            self.shortcut = None
        else:
            self.dwconv = nn.Sequential(
                ConvBNReLU(in_dim, mid_channel, kernel_size=3, stride=stride, padding=1, groups=in_dim, bias=False),
                ConvBNReLU(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=mid_channel, bias=False)
            )
            self.shortcut = nn.Sequential(
                ConvModule(in_dim, in_dim, kernel_size=3, stride=stride, padding=1, groups=in_dim, norm_cfg=dict(type='BatchNorm2d'), act_cfg=None, bias=False),
                ConvModule(in_dim, out_dim, kernel_size=1, stride=1, padding=0, norm_cfg=dict(type='BatchNorm2d'), act_cfg=None, bias=False)
            )

        self.conv2 = ConvModule(mid_channel, out_dim, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=dict(type='BatchNorm2d'), act_cfg=None)
        return

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        return F.relu(x + identity)


class SemanticBranch(torch.nn.Module):
    def __init__(self, in_channels: int = 3,
                 expand: int = 6,
                 semantic_channels: Tuple[int] = (16, 32, 64, 128)):
        super(SemanticBranch, self).__init__()
        self.stages = list()
        for i in range(len(semantic_channels)):
            stage_name = f'stage{i + 1}'
            self.stages.append(stage_name)
            if i == 0:
                self.add_module(stage_name, StemBlock(in_channels, semantic_channels[i]))
            elif i == len(semantic_channels) - 1:
                self.add_module(stage_name,
                                nn.Sequential(
                                    GatherAndExpansionLayer(semantic_channels[i - 1], semantic_channels[i], expand, 2),
                                    GatherAndExpansionLayer(semantic_channels[i], semantic_channels[i], expand, 1),
                                    GatherAndExpansionLayer(semantic_channels[i], semantic_channels[i], expand, 1),
                                    GatherAndExpansionLayer(semantic_channels[i], semantic_channels[i], expand, 1),
                                )
                                )
            else:
                self.add_module(stage_name,
                                nn.Sequential(
                                    GatherAndExpansionLayer(semantic_channels[i - 1], semantic_channels[i], expand, 2),
                                    GatherAndExpansionLayer(semantic_channels[i], semantic_channels[i], expand, 1)
                                )
                                )

        self.add_module('ce', ContextEmbeddingBlock(semantic_channels[-1], semantic_channels[-1]))
        self.stages.append('ce')
        return

    def forward(self, x):
        semantic_outs = list()
        for stage in self.stages:
            x = getattr(self, stage)(x)
            semantic_outs.append(x)
        return semantic_outs


class DetailBranch(torch.nn.Module):
    def __init__(self, in_channels: int = 3,
                 detail_channels: Tuple[int] = (64, 64, 128)):
        super(DetailBranch, self).__init__()
        detail_branch = list()
        for i in range(len(detail_channels)):
            if i == 0:
                detail_branch.append(
                    nn.Sequential(
                        ConvBNReLU(in_channels=in_channels,
                                   out_channels=detail_channels[i],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                   ),
                        ConvBNReLU(in_channels=detail_channels[i],
                                   out_channels=detail_channels[i],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False),
                    )
                )
            else:
                detail_branch.append(
                    nn.Sequential(
                        ConvBNReLU(in_channels=detail_channels[i - 1],
                                   out_channels=detail_channels[i],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                        ConvBNReLU(in_channels=detail_channels[i],
                                   out_channels=detail_channels[i],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False),
                        ConvBNReLU(in_channels=detail_channels[i],
                                   out_channels=detail_channels[i],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False),
                    )
                )

        self.detail_branch = nn.ModuleList(detail_branch)
        return

    def forward(self, x):
        for stage in self.detail_branch:
            x = stage(x)
        return x


class BilateralGuidedAggregation(torch.nn.Module):
    def __init__(self, dim, align_corners):
        super(BilateralGuidedAggregation, self).__init__()
        self.detail_dwconv = torch.nn.Sequential(
            ConvModule(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, norm_cfg=dict(type='BatchNorm2d'), act_cfg=None, bias=False),
            ConvModule(dim, dim, kernel_size=1, stride=1, padding=0, norm_cfg=None, act_cfg=None, bias=False)
        )

        self.detail_down = torch.nn.Sequential(
            ConvModule(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False, norm_cfg=dict(type='BatchNorm2d'), act_cfg=None),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.semantic_conv = nn.Sequential(
            ConvModule(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BatchNorm2d'),
                act_cfg=None)
        )

        self.semantic_dwconv = torch.nn.Sequential(
            ConvModule(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, norm_cfg=dict(type='BatchNorm2d'), act_cfg=None, bias=False),
            ConvModule(dim, dim, kernel_size=1, stride=1, padding=0, norm_cfg=None, act_cfg=None, bias=False),
        )

        self.conv = ConvModule(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, norm_cfg=dict(type='BatchNorm2d'), act_cfg=dict(type='ReLU'), bias=False)

        self.align_corners = align_corners
        return

    def forward(self, dfm, sfm):
        detail_dwconv = self.detail_dwconv(dfm)
        detail_down = self.detail_down(dfm)
        semantic_conv = self.semantic_conv(sfm)
        semantic_dwconv = self.semantic_dwconv(sfm)

        semantic_conv = F.interpolate(semantic_conv, detail_dwconv.shape[2:], mode='bilinear', align_corners=self.align_corners)

        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        fuse_2 = detail_down * torch.sigmoid(semantic_dwconv)

        fuse_2 = F.interpolate(fuse_2, fuse_1.shape[2:], mode='bilinear', align_corners=self.align_corners)

        return self.conv(fuse_1 + fuse_2)


@BUILD_NETWORK_REGISTRY.register()
class BiSeNetV2(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 detail_channels: Tuple[int] = (64, 64, 128),
                 semantic_channels: Tuple[int] = (16, 32, 64, 128),
                 bga_channels: int = 128,
                 semantic_expansion_ratio=6,
                 out_indices: Tuple[int] = (0, 1, 2, 3, 4),
                 align_corners=False):
        super(BiSeNetV2, self).__init__()

        self.detail = DetailBranch(in_channels, detail_channels)
        self.semantic = SemanticBranch(in_channels, semantic_expansion_ratio, semantic_channels)
        self.bga = BilateralGuidedAggregation(bga_channels, align_corners)

        self.out_indices = out_indices
        return

    def forward(self, x):
        x_detail = self.detail(x)
        x_semantic = self.semantic(x)
        x_head = self.bga(x_detail, x_semantic[-1])
        outs = [x_head] + x_semantic[:-1]
        return tuple([outs[i] for i in self.out_indices])
