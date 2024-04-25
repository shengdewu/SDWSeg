from typing import Type, Union, List, Optional, Dict, Tuple

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from engine.model.build import BUILD_NETWORK_REGISTRY

from ..utils import ConvModule

__all__ = [
    'BasicBlock',
    'Bottleneck',
    'ResNet'
]


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_cfg: Optional[Dict] = None
    ) -> None:
        super(BasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BatchNorm2d')

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvModule(inplanes, planes,
                                padding=1, kernel_size=3, stride=stride, groups=1, bias=False, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

        self.conv2 = ConvModule(planes, planes,
                                padding=1, kernel_size=3, stride=1, groups=1, bias=False, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=None)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_cfg: Optional[Dict] = None
    ) -> None:
        super(Bottleneck, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BatchNorm2d')

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvModule(inplanes, width,
                                padding=0, kernel_size=1, stride=1, bias=False,
                                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

        self.conv2 = ConvModule(width, width,
                                padding=dilation, kernel_size=3, stride=stride, groups=groups, bias=False, dilation=dilation,
                                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

        self.conv3 = ConvModule(width, planes * self.expansion,
                                padding=0, kernel_size=1, stride=1, bias=False,
                                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        return

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


@BUILD_NETWORK_REGISTRY.register()
class ResNet(nn.Module):
    """ResNet/ResNext backbone.

    resnext50_32x4d : dict(depth=50, groups=32, width_per_group=4)
    resnext101_32x8d : dict(depth=101, groups=32, width_per_group=8)
    wide_resnet50_2 : dict(depth=50, groups=1, width_per_group=128)
    wide_resnet101_2 : dict(depth=101, groups=1, width_per_group=128)
    resnext50_32x4d : dict(depth=50, groups=32, width_per_group=4)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(
            self,
            depth: int,
            in_channels: int = 3,
            res_channels: int = 64,
            dilation: int = 1,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_cfg: Optional[Dict] = None,
            out_indices: List[int] = (0, 1, 2, 3, 4),
            stem_kernel: int = 7,
            stem_sharp: bool = True
    ) -> None:
        super(ResNet, self).__init__()

        if depth not in self.arch_settings.keys():
            raise KeyError(f'invalid depth {depth} for resnet/resnext')
        arch = self.arch_settings[depth]

        if norm_cfg is None:
            norm_cfg = dict(type='BatchNorm2d')
        self.norm_cfg = norm_cfg
        self.act_cfg = dict(type='ReLU', inplace=True)

        self.inplanes = res_channels
        self.dilation = dilation
        self.out_indices = out_indices

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        stem = [
            ConvModule(
                in_channels,
                res_channels,
                kernel_size=stem_kernel, stride=2, padding=stem_kernel // 2, bias=False, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        ]
        if stem_sharp:
            stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.stem = nn.Sequential(*stem)

        self.layers = nn.ModuleList()

        self.layers.append(self._make_layer(arch[0], res_channels, arch[1][0]))
        self.layers.append(self._make_layer(arch[0], res_channels * 2, arch[1][1], stride=2,
                                            dilate=replace_stride_with_dilation[0]))
        self.layers.append(self._make_layer(arch[0], res_channels * 4, arch[1][2], stride=2,
                                            dilate=replace_stride_with_dilation[1]))
        self.layers.append(self._make_layer(arch[0], res_channels * 8, arch[1][3], stride=2,
                                            dilate=replace_stride_with_dilation[2]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_cfg = self.norm_cfg
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvModule(self.inplanes,
                                    planes * block.expansion,
                                    padding=0,
                                    kernel_size=1,
                                    stride=stride,
                                    norm_cfg=self.norm_cfg,
                                    bias=False,
                                    act_cfg=None)

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, self.groups,
                                self.base_width, self.dilation, norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.stem(x)
        outs = list()
        if 0 in self.out_indices:
            outs.append(x)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i + 1 in self.out_indices:
                outs.append(x)
        return tuple(outs)
