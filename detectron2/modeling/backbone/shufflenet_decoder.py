#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from detectron2.layers import get_norm
from detectron2.layers.deform_conv_hw import (
    DeformConvBound,
    DeformConvOriginal,
    DeformConvOriginalDepthWise,
    DeformConvSquare,
    DeformConvSquareDepthWise,
)


def get_conv_func(norm_type, conv_func_type, in_channels, out_channels, init_type):

    if conv_func_type == "k1":
        conv_func = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
    elif conv_func_type == "k3":
        conv_func = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
        )
    elif conv_func_type == "k3dw":
        conv_func = DepthWiseConv2d(norm_type, in_channels, out_channels)
    elif conv_func_type == "deform_conv_original":
        conv_func = DeformConvOriginal(in_channels, out_channels)
    elif conv_func_type == "deform_conv_bound":
        conv_func = DeformConvBound(in_channels, out_channels)
    elif conv_func_type == "deform_conv_square":
        conv_func = DeformConvSquare(in_channels, out_channels)
    elif conv_func_type == "deform_conv_original_depthwise":
        conv_func = DeformConvOriginalDepthWise(in_channels, out_channels)
    elif conv_func_type == "deform_conv_square_depthwise":
        conv_func = DeformConvSquareDepthWise(in_channels, out_channels)

    if init_type == "kaiming_fan_in_a0":
        nn.init.kaiming_normal_(conv_func.weight, a=0, mode="fan_in", nonlinearity="relu")
    elif init_type == "kaiming_fan_out_a0":
        nn.init.kaiming_normal_(conv_func.weight, a=0, mode="fan_out", nonlinearity="relu")
    elif init_type == "kaiming_fan_out_a1":
        nn.init.kaiming_normal_(conv_func.weight, a=1, mode="fan_out", nonlinearity="leaky_relu")
    elif init_type == "kaiming_fan_out_a1_1x1":
        nn.init.normal_(conv_func.weight, mean=0, std=math.sqrt(1.0 / out_channels))
    elif init_type == "kaiming_fan_out_a0_1x1":
        nn.init.normal_(conv_func.weight, mean=0, std=math.sqrt(2.0 / out_channels))
    elif init_type == "kaiming_fan_out_a1_3x3":
        nn.init.normal_(conv_func.weight, mean=0, std=math.sqrt(1.0 / out_channels) / 3.0)
    elif init_type == "kaiming_fan_out_a0_3x3":
        nn.init.normal_(conv_func.weight, mean=0, std=math.sqrt(2.0 / out_channels) / 3.0)

    return conv_func


class DepthWiseConv2d(nn.Module):
    def __init__(
        self, norm_type, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1
    ):
        super(DepthWiseConv2d, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.point_wise = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            get_norm(norm_type, out_channel),
            nn.ReLU(inplace=True),
        )

        self.kernel_size = _pair(kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channel, 1, *self.kernel_size))
        self.bias = None
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = out_channel

    def forward(self, x):
        x = self.point_wise(x)
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channel)
        tmpstr += ", out_channels=" + str(self.out_channel)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", bias=False"
        return tmpstr


def MultiConvsList(conv_type, init_type, num_convs, in_channel, out_channel, norm_type):

    if num_convs < 1:
        assert in_channel == out_channel
        return [nn.Identity()]

    list_convs = [
        get_conv_func(norm_type, conv_type, in_channel, out_channel, init_type),
        get_norm(norm_type, out_channel),
        nn.ReLU(inplace=True),
    ]

    for _ in range(num_convs - 1):
        list_convs.extend(
            [
                get_conv_func(norm_type, conv_type, in_channel, out_channel, init_type),
                get_norm(norm_type, out_channel),
                nn.ReLU(inplace=True),
            ]
        )

    return list_convs


class MultiConvsHeads(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg):
        super(MultiConvsHeads, self).__init__()
        # fmt: off
        num_classes = cfg.NUM_CLASSES
        in_channel  = cfg.DECODER.DIM
        out_channel = cfg.HEAD.DIM
        norm_type   = cfg.HEAD.NORM
        init_type   = cfg.HEAD.INIT
        conv_type   = cfg.HEAD.CONV
        num_convs   = cfg.HEAD.NUM_CONVS
        prior_prob  = cfg.HEAD.PRIOR_PROB
        # fmt: on

        if num_convs < 1:
            assert in_channel == out_channel

        list_cls_feat = MultiConvsList(
            conv_type, init_type, num_convs, in_channel, out_channel, norm_type
        )
        self.cls_feat = nn.Sequential(*list_cls_feat)
        self.cls_head = nn.Conv2d(out_channel, num_classes, kernel_size=1)

        list_wh_feat = MultiConvsList(
            conv_type, init_type, num_convs, in_channel, out_channel, norm_type
        )
        self.wh_feat = nn.Sequential(*list_wh_feat)
        self.wh_head = nn.Conv2d(out_channel, 2, kernel_size=1)

        list_reg_feat = MultiConvsList(
            conv_type, init_type, num_convs, in_channel, out_channel, norm_type
        )
        self.reg_feat = nn.Sequential(*list_reg_feat)
        self.reg_head = nn.Conv2d(out_channel, 2, kernel_size=1)

        for modules in [self.wh_head, self.reg_head]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_head.bias, bias_value)

    def forward(self, x):
        pred_cls = self.cls_head(self.cls_feat(x))
        pred_wh = self.wh_head(self.wh_feat(x))
        pred_reg = self.reg_head(self.reg_feat(x))

        return {"cls": pred_cls, "wh": pred_wh, "reg": pred_reg}


def build_convs_stack(norm_type, channels, conv_func, init_type):
    stack_list = []
    for i in range(len(channels) - 1):
        stack_list.extend(
            [
                get_conv_func(norm_type, conv_func, channels[i], channels[i + 1], init_type),
                get_norm(norm_type, channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            ]
        )
    return nn.Sequential(*stack_list)


def build_shufflenet_decoder(cfg):
    # fmt: off
    channels    = cfg.DECODER.CHANNELS
    channels.append(cfg.DECODER.DIM)
    norm_type   = cfg.DECODER.NORM
    init_type   = cfg.DECODER.INIT
    conv_func   = cfg.DECODER.CONV
    # fmt: on

    return build_convs_stack(norm_type, channels, conv_func, init_type), MultiConvsHeads(cfg)
