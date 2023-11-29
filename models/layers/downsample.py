import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmcv.cnn import ConvModule, build_norm_layer

from timm.models.layers import to_2tuple, LayerNorm2d

from .blur_pool import apply_blurpool


if "LayerNorm2d" not in MODELS:
    MODELS.register_module("LayerNorm2d", module=LayerNorm2d)


class DownsampleV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size,
        kernel_size,
        norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        img_size=224,
    ):
        super().__init__()

        assert patch_size in (2, 4)
        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        if patch_size <= kernel_size:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=patch_size,
                padding=(kernel_size - 1) // 2,
            )
        else:
            dim = out_channels // 2
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.Conv2d(
                    out_channels // 2,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=patch_size // 2,
                    padding=(kernel_size - 1) // 2,
                ),
            )

        self.norm = (
            build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg else nn.Identity()
        )

    def forward(self, x):
        # x: B C H W
        x = self.proj(x)
        x = self.norm(x)
        return x


class DownsampleV2(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        img_size=224,
        kernel_size=3,
        patch_size=4,
        ratio=0.5,
        conv_cfg=None,
        conv_bias=True,
        norm_cfg=dict(type="LayerNorm2d"),
        act_cfg=dict(type="GELU"),
        with_blurpool=False,
        order=("conv", "norm", "act"),
        **kwargs
    ):
        super().__init__()
        assert patch_size in (2, 4)

        img_size = to_2tuple(img_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        if patch_size == 4:
            mid_chs = int(out_chs * ratio)
            self.conv1 = ConvModule(
                in_chs,
                mid_chs,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
                bias=conv_bias,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order,
            )
        else:
            mid_chs = in_chs
            self.conv1 = nn.Identity()

        self.conv2 = ConvModule(
            mid_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size - 1) // 2,
            bias=conv_bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            order=order,
        )
        if with_blurpool:
            apply_blurpool(self.conv1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


downsampler_cfg = {
    # layer_abbreviation: module
    "DownsampleV1": DownsampleV1,
    "DownsampleV2": DownsampleV2,
}


def build_downsample_layer(cfg):
    """Build downsample (stem or transition) layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify activation layer type.
            layer args: args needed to instantiate a stem layer.

    Returns:
        layer (nn.Module): Created stem layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in downsampler_cfg:
        raise KeyError("Unrecognized stem type {}".format(layer_type))
    else:
        layer = downsampler_cfg[layer_type]
        if layer is None:
            raise NotImplementedError

    layer = layer(**cfg_)
    return layer
