# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from functools import lru_cache
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from .batch_norm import get_norm
from .deform_conv import deform_conv
from .wrappers import _NewEmptyTensorOp


class DeformConvOriginal(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
    ):
        super(DeformConvOriginal, self).__init__()

        assert not bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        self.bias = None

        self.conv_offset = nn.Conv2d(
            in_channels, kernel_size * kernel_size * 2 * deformable_groups, kernel_size=1, padding=0
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        o = self.conv_offset(x)
        x = deform_conv(
            x,
            o,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr


class DeformConvBound(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        offset_bound=8,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
    ):
        super(DeformConvBound, self).__init__()

        assert not bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        self.bias = None

        self.conv_offset = nn.Conv2d(
            in_channels, kernel_size * kernel_size * 2 * deformable_groups, kernel_size=3, padding=1
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

        self.offset_min, self.offset_max = -offset_bound, offset_bound

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        o = self.conv_offset(x)
        x = deform_conv(
            x,
            o.clamp_(self.offset_min, self.offset_max),
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr


class DeformConvSquare(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        offset_bound=8,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
    ):
        super(DeformConvSquare, self).__init__()

        assert not bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset_bound = offset_bound
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        self.bias = None

        self.conv_offset = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, padding=1)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

        self.register_buffer(
            "offset_template",
            torch.FloatTensor([-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1])
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(2),
        )

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        s = self.conv_offset(x)
        o = self.offset_template * s.clamp_(0, self.offset_bound)
        x = deform_conv(
            x,
            o,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr


class DeformConvSquareDepthWise(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        offset_bound=8,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
        bias=False,
    ):
        super(DeformConvSquareDepthWise, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset_bound = offset_bound
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = out_channels
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size, kernel_size))
        self.bias = None

        self.conv_offset = nn.Conv2d(out_channels, deformable_groups, kernel_size=1, padding=0)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

        self.point_wise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.register_buffer(
            "offset_template",
            torch.FloatTensor([-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1])
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(2),
        )

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = self.point_wise(x)
        s = self.conv_offset(x)
        o = self.offset_template * s.clamp_(0, self.offset_bound)
        x = deform_conv(
            x,
            o,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr


class DeformConvOriginalDepthWise(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
        bias=False,
    ):
        super(DeformConvOriginalDepthWise, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = out_channels
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size, kernel_size))
        self.bias = None

        self.conv_offset = nn.Conv2d(
            out_channels,
            kernel_size * kernel_size * 2 * deformable_groups,
            kernel_size=1,
            padding=0,
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

        self.point_wise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = self.point_wise(x)
        o = self.conv_offset(x)
        x = deform_conv(
            x,
            o,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr
