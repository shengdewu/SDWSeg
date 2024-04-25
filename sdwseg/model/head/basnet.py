import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from ..backbone import BasicBlock

from engine.model import BUILD_NETWORK_REGISTRY

from .base_head import BaseHead
from ..utils import ConvModule

__all__ = [
    'BASNet'
]


class RefineNet(nn.Module):
    def __init__(self, in_channels, head_channels, refine_channels=64, num_stages=4):
        super(RefineNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.stem = nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, stride=1, bias=False)

        for i in range(num_stages):
            if i == 0:
                in_ch = head_channels
                out_ch = refine_channels
            else:
                in_ch = refine_channels
                out_ch = refine_channels

            self.encoder.append(
                nn.Sequential(
                    ConvModule(in_ch,
                               out_ch,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               norm_cfg=dict(type='BatchNorm2d'),
                               act_cfg=dict(type='ReLU', inplace=True)),
                    nn.MaxPool2d(2, 2, ceil_mode=True)
                )
            )
            self.decoder.append(
                nn.Sequential(
                    ConvModule(refine_channels * 2,
                               refine_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               norm_cfg=dict(type='BatchNorm2d'),
                               act_cfg=dict(type='ReLU', inplace=True)),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
            )

        self.bottle = ConvModule(refine_channels,
                                 refine_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1,
                                 norm_cfg=dict(type='BatchNorm2d'),
                                 act_cfg=dict(type='ReLU', inplace=True))

        self.head = ConvModule(refine_channels,
                               in_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               norm_cfg=dict(type='BatchNorm2d'),
                               act_cfg=dict(type='ReLU', inplace=True))
        return

    def forward(self, x):
        hx = self.stem(x)
        enc_outs = []
        for enc in self.encoder:
            hx = enc(hx)
            enc_outs.append(hx)

        hx = self.bottle(hx)

        for i in reversed(range(len(self.decoder))):
            hx = self.decoder[i](torch.cat((enc_outs[i], hx), dim=1))

        return x + self.head(hx)


@BUILD_NETWORK_REGISTRY.register()
class BASNet(BaseHead):
    """
    BASNet: Boundary-Aware Salient Object Detection
    """

    def __init__(self, in_channels: List[int],
                 num_classes: int,
                 down_backbone_nums: int = 2,
                 refine_channels: int = 64,
                 refine_head: int = 32,
                 interpolation: str = 'bilinear',
                 align_corners: bool = False,
                 upsample_factor: int = 2,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = None,
                 loss_cfg: Optional[Dict] = None,
                 ignore_index: Optional[int] = None):

        super(BASNet, self).__init__(num_classes, loss_cfg, ignore_index)

        self.interpolation = interpolation
        self.align_corners = align_corners
        if norm_cfg is None:
            norm_cfg = dict(type='BatchNorm2d')

        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)

        self.down_backbone_stage = nn.ModuleList()
        backbone_last_channels = in_channels[-1]
        backbone_channels = in_channels
        for i in range(down_backbone_nums):
            self.down_backbone_stage.append(
                nn.Sequential(
                    nn.MaxPool2d(2, 2, ceil_mode=True),
                    BasicBlock(backbone_last_channels, backbone_last_channels),
                    BasicBlock(backbone_last_channels, backbone_last_channels),
                    BasicBlock(backbone_last_channels, backbone_last_channels)
                ))
            backbone_channels.append(backbone_last_channels)

        self.bottle = nn.Sequential(
            ConvModule(backbone_last_channels, backbone_last_channels, kernel_size=3, padding=2, dilation=2, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(backbone_last_channels, backbone_last_channels, kernel_size=3, padding=2, dilation=2, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(backbone_last_channels, backbone_last_channels, kernel_size=3, padding=2, dilation=2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        self.decoder = nn.ModuleList()
        for i in range(len(backbone_channels)):
            if i == 0:
                self.decoder.append(
                    nn.Sequential(
                        ConvModule(backbone_channels[i] * 2, backbone_channels[i], kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                        ConvModule(backbone_channels[i], backbone_channels[i], kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                        ConvModule(backbone_channels[i], backbone_channels[i], kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ConvModule(backbone_channels[i] * 2, backbone_channels[i-1], kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                        ConvModule(backbone_channels[i-1], backbone_channels[i-1], kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                        ConvModule(backbone_channels[i-1], backbone_channels[i-1], kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                        nn.Upsample(scale_factor=upsample_factor, mode=interpolation, align_corners=align_corners)
                    )
                )

        self.head = nn.Conv2d(backbone_channels[0], self.num_classes, kernel_size=3, padding=1, stride=1)

        self.refine_net = RefineNet(self.num_classes, refine_head, refine_channels=refine_channels)
        return

    def forward(self, x_stages, shape):
        hx = x_stages[-1]
        stages = list()
        for enc in self.down_backbone_stage:
            hx = enc(hx)
            stages.append(hx)

        hx = self.bottle(hx)
        stages = x_stages + tuple(stages)
        for i in reversed(range(len(self.decoder))):
            stage = stages[i]
            if hx.shape[2:] != stage.shape[2:]: # 推荐原图大小是32的偶数倍，否则转onnx时比较麻烦
                stage = F.interpolate(stage, hx.shape[2:], mode=self.interpolation, align_corners=self.align_corners)
            hx = self.decoder[i](torch.cat((hx, stage), dim=1))

        hx = self.head(hx)

        out = self.refine_net(hx)

        if out.shape[2:] != shape:
            out = F.interpolate(out, shape, mode=self.interpolation, align_corners=self.align_corners)
        return out


