from typing import Union, List

import torch
import torch.nn as nn

from engine.model import BUILD_NETWORK_REGISTRY

from ..utils import ConvBNReLU


__all__ = [
    'STDCBackbone'
]


class STDCBlock(nn.Module):
    """
    STDC building block, known as Short Term Dense Concatenate module.
    In STDC module, the kernel size of first block is 1, and the rest of them are simply set as 3.
    """

    def __init__(self, in_channels: int, out_channels: int, steps: int, stdc_downsample_mode: str, stride: int):
        """
        :param steps: The total number of convs in this module, 1 conv 1x1 and (steps - 1) conv3x3.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        """
        super().__init__()
        if steps not in [2, 3, 4]:
            raise ValueError(f"only 2, 3, 4 steps number are supported, found: {steps}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.steps = steps
        self.stdc_downsample_mode = stdc_downsample_mode
        self.stride = stride
        self.conv_list = nn.ModuleList()
        # build first step conv 1x1.
        self.conv_list.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, bias=False))
        # build skip connection after first convolution.
        if stride == 1:
            self.skip_step1 = nn.Identity()
        elif stdc_downsample_mode == "avg_pool":
            self.skip_step1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif stdc_downsample_mode == "dw_conv":
            self.skip_step1 = ConvBNReLU(
                out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False, groups=out_channels // 2, use_activation=False
            )
        else:
            raise ValueError(f"stdc_downsample mode is not supported: found {stdc_downsample_mode}," f" must be in [avg_pool, dw_conv]")

        in_channels = out_channels // 2
        mid_channels = in_channels
        # build rest conv3x3 layers.
        for idx in range(1, steps):
            if idx < steps - 1:
                mid_channels //= 2
            conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_list.append(conv)
            in_channels = mid_channels

        # add dw conv before second step for down sample if stride = 2.
        if stride == 2:
            self.conv_list[1] = nn.Sequential(
                ConvBNReLU(
                    out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, groups=out_channels // 2, use_activation=False, bias=False
                ),
                self.conv_list[1],
            )

    def forward(self, x):
        out_list = []
        # run first conv
        x = self.conv_list[0](x)
        out_list.append(self.skip_step1(x))

        for conv in self.conv_list[1:]:
            x = conv(x)
            out_list.append(x)

        out = torch.cat(out_list, dim=1)
        return out


@BUILD_NETWORK_REGISTRY.register()
class STDCBackbone(nn.Module):
    def __init__(
            self,
            block_types: list,
            ch_widths: list,
            num_blocks: list,
            stdc_steps: int = 4,
            stdc_downsample_mode: str = "avg_pool",
            in_channels: int = 3,
            out_indices: List[int] = (2, 3, 4)
    ):
        """
        Rethinking BiSeNet For Real-time Semantic Segmentation

        :param block_types: list of block type for each stage, supported `conv` for ConvBNRelu with 3x3 kernel.
        :param ch_widths: list of output num of channels for each stage.
        :param num_blocks: list of the number of repeating blocks in each stage.
        :param stdc_steps: num of convs steps in each block.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :param in_channels: num channels of the input image.
        :param out_indices: the idx of output
        """
        super(STDCBackbone, self).__init__()
        if not (len(block_types) == len(ch_widths) == len(num_blocks)):
            raise ValueError(
                f"STDC architecture configuration, block_types, ch_widths, num_blocks, must be defined for the same number"
                f" of stages, found: {len(block_types)} for block_type, {len(ch_widths)} for ch_widths, "
                f"{len(num_blocks)} for num_blocks"
            )

        self.out_widths = []
        self.stages = nn.ModuleDict()
        self.out_stage_keys = []
        down_ratio = 2
        for i, (block_type, width, blocks) in enumerate(zip(block_types, ch_widths, num_blocks)):
            block_name = f"block_s{down_ratio}"
            self.stages[block_name] = self._make_stage(
                in_channels=in_channels,
                out_channels=width,
                block_type=block_type,
                num_blocks=blocks,
                stdc_steps=stdc_steps,
                stdc_downsample_mode=stdc_downsample_mode,
            )
            if i in out_indices:
                self.out_stage_keys.append(block_name)
            in_channels = width
            down_ratio *= 2
        return

    def _make_stage(self, in_channels: int, out_channels: int, block_type: str, num_blocks: int, stdc_downsample_mode: str, stdc_steps: int = 4):
        """
        :param in_channels: input channels of stage.
        :param out_channels: output channels of stage.
        :param block_type: stage building block, supported `conv` for 3x3 ConvBNRelu, or `stdc` for STDCBlock.
        :param num_blocks: num of blocks in each stage.
        :param stdc_steps: number of conv3x3 steps in each STDC block, referred as `num blocks` in paper.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :return: nn.Module
        """
        if block_type == "conv":
            block = ConvBNReLU
            kwargs = {"kernel_size": 3, "padding": 1, "bias": False}
        elif block_type == "stdc":
            block = STDCBlock
            kwargs = {"steps": stdc_steps, "stdc_downsample_mode": stdc_downsample_mode}
        else:
            raise ValueError(f"Block type not supported: {block_type}, excepted: `conv` or `stdc`")

        # first block to apply stride 2.
        blocks = nn.ModuleList([block(in_channels, out_channels, stride=2, **kwargs)])
        # build rest of blocks
        for i in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels, stride=1, **kwargs))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        for stage_name, stage in self.stages.items():
            x = stage(x)
            if stage_name in self.out_stage_keys:
                outputs.append(x)
        return tuple(outputs)
