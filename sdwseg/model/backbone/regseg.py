from typing import List

import torch
import torch.nn as nn

from engine.model import BUILD_NETWORK_REGISTRY

from ..utils import SELayer, ConvBNReLU

__all__ = [
    'RegSegEncoder'
]


class AdaptiveShortcutBlock(nn.Module):
    """
    Adaptive shortcut makes the following adaptations, if needed:
    Applying pooling if stride > 1
    Applying 1x1 conv if in/out channels are different or if pooling was applied
    If stride is 1 and in/out channels are the same, then the shortcut is just an identity
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        shortcut_layers = [nn.Identity()]
        if stride != 1:
            shortcut_layers[0] = nn.AvgPool2d(stride, stride, ceil_mode=True)  # override the identity layer
        if in_channels != out_channels or stride != 1:
            shortcut_layers.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, bias=False, use_activation=False))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        return self.shortcut(x)


class SplitDilatedGroupConvBlock(nn.Module):
    """
    Splits the input to "dilation groups", following grouped convolution with different dilation for each group
    """

    def __init__(self, in_channels: int, split_dilations: List[int], group_width_per_split: int, stride: int, bias: bool):
        """
        :param split_dilations:         a list specifying the required dilations.
                                        the input will be split into len(dilations) groups,
                                        group [i] will be convolved with grouped dilated (dilations[i]) convolution
        :param group_width_per_split:   the group width for the *inner* dilated convolution
        """
        super().__init__()
        self.num_splits = len(split_dilations)
        assert in_channels % self.num_splits == 0, f"Cannot split {in_channels} to {self.num_splits} groups with equal size."
        group_channels = in_channels // self.num_splits
        assert group_channels % group_width_per_split == 0, (
            f"Cannot split {group_channels} channels ({in_channels} / {self.num_splits} splits)" f" to groups with {group_width_per_split} channels per group."
        )
        inner_groups = group_channels // group_width_per_split
        self.convs = nn.ModuleList(
            nn.Conv2d(group_channels, group_channels, 3, padding=d, dilation=d, stride=stride, bias=bias, groups=inner_groups) for d in split_dilations
        )
        self._splits = [in_channels // self.num_splits] * self.num_splits

    def forward(self, x):
        x = torch.split(x, self._splits, dim=1)
        return torch.cat([self.convs[i](x[i]) for i in range(self.num_splits)], dim=1)


class DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int], group_width: int, stride: int, se_ratio: int = 4):
        """
        :param dilations:           a list specifying the required dilations.
                                    the input will be split into len(dilations) groups,
                                    group [i] will be convolved with grouped dilated (dilations[i]) convolution
        :param group_width:         the group width for the dilated convolution(s)
        :param se_ratio:            the ratio of the squeeze-and-excitation block w.r.t in_channels (as in the paper)
                                    for example: a value of 4 translates to in_channels // 4
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.group_width = group_width
        self.stride = stride
        self.se_ratio = se_ratio
        self.shortcut = AdaptiveShortcutBlock(in_channels, out_channels, stride)
        groups = out_channels // group_width

        if len(dilations) == 1:  # minor optimization: no need to split if we only have 1 dilation group
            dilation = dilations[0]
            dilated_conv = nn.Conv2d(out_channels, out_channels, 3, stride=stride, groups=groups, padding=dilation, dilation=dilation, bias=False)
        else:
            dilated_conv = SplitDilatedGroupConvBlock(out_channels, dilations, group_width_per_split=group_width, stride=stride, bias=False)

        self.d_block_path = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=1, bias=False),
            dilated_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # the ratio of se block applied to `in_channels` as in the original paper
            SELayer(out_channels, in_channels // se_ratio),
            ConvBNReLU(out_channels, out_channels, 1, use_activation=False, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.d_block_path(x)
        out = self.relu(x1 + x2)
        return out

    def __str__(self):
        return (
            f"{self.__class__.__name__}_in{self.in_channels}_out{self.out_channels}" f"_d{self.dilations}_gw{self.group_width}_s{self.stride}_se{self.se_ratio}"
        )


@BUILD_NETWORK_REGISTRY.register()
class RegSegEncoder(nn.Module):
    def __init__(self, stages: List[List], out_indices=(0, 1, 2), stem_channels: int = 32):
        super().__init__()
        self.stem = ConvBNReLU(in_channels=3, out_channels=stem_channels, kernel_size=3, stride=2, padding=1)
        self.stages = self._generate_stages(stem_channels, stages)
        self.out_indices = out_indices
        return

    @staticmethod
    def _generate_stages(in_channels, stages_cfg: List[List]) -> nn.ModuleList:
        prev_out_channels = in_channels
        stages = nn.ModuleList()
        for stage in stages_cfg:
            stage_blocks = nn.Sequential()
            for i, (out_channels, dilations, group_width, stride, se_ratio) in enumerate(stage):
                d_block = DBlock(prev_out_channels, out_channels, dilations, group_width, stride, se_ratio)
                prev_out_channels = d_block.out_channels
                stage_blocks.add_module(f"{str(d_block)}#{i}", d_block)  # NOTE: {i} distinguishes blocks with same name
            stages.append(stage_blocks)
        return stages

    def forward(self, x):
        outputs = list()
        x_in = self.stem(x)
        for stage in self.stages:
            x_out = stage(x_in)
            outputs.append(x_out)
            x_in = x_out  # last stage out is next stage in
        return tuple([outputs[i] for i in self.out_indices])
