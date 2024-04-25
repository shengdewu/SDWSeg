from typing import Union, Tuple, Type, Optional, Dict
from torch import nn
import importlib

__all__ = [
    'ConvBNReLU',
    'ConvBNAct',
    'ConvModule'
]


def build_norm(num_features: int, kwargs: Dict, num_groups: Optional[int] = None):
    normal_module = 'torch.nn'
    module = importlib.import_module(normal_module)
    param = dict()

    norm_type = ''
    for key, value in kwargs.items():
        if key.lower() == 'type':
            norm_type = value
            continue
        param[key.lower()] = value

    if norm_type == 'GroupNorm':
        assert num_groups is not None
        norm = getattr(module, norm_type)(num_groups=num_groups, num_channels=num_features, **param)
    else:
        norm = getattr(module, norm_type)(num_features, **param)
    return norm


def build_act(kwargs: Dict):
    normal_module = 'torch.nn'
    module = importlib.import_module(normal_module)
    param = dict()
    for key, value in kwargs.items():
        if key == 'type':
            continue
        param[key] = value

    return getattr(module, kwargs['type'])(**param)


class Residual(nn.Module):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a residual skip connection within a single block.
    """

    def forward(self, x):
        return x


def autopad(kernel, padding=None):
    # PAD TO 'SAME'
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


class ConvBNAct(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Activation layer.
        Default behaviour is Conv-BN-Act. To exclude Batchnorm module use
        `use_normalization=False`, to exclude activation use `activation_type=None`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int]],
            activation_type: Type[nn.Module],
            stride: Union[int, Tuple[int, int]] = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            use_normalization: bool = True,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            activation_kwargs=None,
    ):

        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {}

        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
        )

        if use_normalization:
            self.seq.add_module(
                "bn",
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype),
            )
        if activation_type is not None:
            self.seq.add_module("act", activation_type(**activation_kwargs))

    def forward(self, x):
        return self.seq(x)


class ConvBNReLU(ConvBNAct):
    """
    Class for Convolution2d-Batchnorm2d-Relu layer. To exclude Batchnorm module use
        `use_normalization=False`, to exclude Relu activation use `use_activation=False`.

    It exists to keep backward compatibility and will be superseeded by ConvBNAct in future releases.
    For new classes please use ConvBNAct instead.

    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            use_normalization: bool = True,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            use_activation: bool = True,
            inplace: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation_type=nn.ReLU if use_activation else None,
            activation_kwargs=dict(inplace=inplace) if inplace else None,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            use_normalization=use_normalization,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )


class ConvModule(nn.Module):
    """
    Class for Convolution2d-norm2d-Activation layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_groups (int): num_groups of GroupNorm
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            norm_cfg: Optional[Dict] = None,
            act_cfg: Optional[Dict] = dict(type='ReLU'),
            norm_groups: Optional[int] = None,
            order=('conv', 'norm', 'act')
    ):
        super().__init__()

        self.seq = nn.Sequential()

        for layer in order:
            if layer == 'conv':
                self.seq.add_module(
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        padding_mode=padding_mode,
                    ),
                )
            elif layer == 'norm' and norm_cfg is not None:
                self.seq.add_module(
                    "bn",
                    build_norm(out_channels, norm_cfg, norm_groups),
                )
            elif layer == 'act' and act_cfg is not None:
                self.seq.add_module("act", build_act(act_cfg))

    def forward(self, x):
        return self.seq(x)
