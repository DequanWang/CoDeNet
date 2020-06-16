import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from . import Conv2d, get_norm

# #TODO: fix lib.gpu.conv_rectify & lib.cpu.conv_rectify

# # copied from https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/functions/rectify.py
# class _rectify(Function):
#     @staticmethod
#     def forward(ctx, y, x, kernel_size, stride, padding, dilation, average):
#         ctx.save_for_backward(x)
#         # assuming kernel_size is 3
#         kernel_size = [k + 2 * (d - 1) for k,d in zip(kernel_size, dilation)]
#         ctx.kernel_size = kernel_size
#         ctx.stride = stride
#         ctx.padding = padding
#         ctx.dilation = dilation
#         ctx.average = average
#         if x.is_cuda:
#             lib.gpu.conv_rectify(y, x, kernel_size, stride, padding, dilation, average)
#         else:
#             lib.cpu.conv_rectify(y, x, kernel_size, stride, padding, dilation, average)
#         ctx.mark_dirty(y)
#         return y

#     @staticmethod
#     def backward(ctx, grad_y):
#         x, = ctx.saved_variables
#         if x.is_cuda:
#             lib.gpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
#                                  ctx.padding, ctx.dilation, ctx.average)
#         else:
#             lib.cpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
#                                  ctx.padding, ctx.dilation, ctx.average)
#         ctx.mark_dirty(grad_y)
#         return grad_y, None, None, None, None, None, None

# rectify = _rectify.apply


# # copied from https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/rectify.py
# class RFConv2d(Conv2d):
#     """Rectified Convolution
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='zeros',
#                  average_mode=False):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         self.rectify = average_mode or (padding[0] > 0 or padding[1] > 0)
#         self.average = average_mode

#         super(RFConv2d, self).__init__(
#                  in_channels, out_channels, kernel_size, stride=stride,
#                  padding=padding, dilation=dilation, groups=groups,
#                  bias=bias, padding_mode=padding_mode)

#     def _conv_forward(self, input, weight):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
#                             weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, input):
#         output = self._conv_forward(input, self.weight)
#         if self.rectify:
#             output = rectify(output, input, self.kernel_size, self.stride,
#                              self.padding, self.dilation, self.average)
#         return output

#     def extra_repr(self):
#         return super().extra_repr() + ', rectify={}, average_mode={}'. \
#             format(self.rectify, self.average)


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size=3,
        stride=(1, 1),
        padding=1,
        dilation=(1, 1),
        groups=1,
        bias=True,
        radix=2,
        reduction_factor=4,
        rectify=False,
        rectify_avg=False,
        norm=None,
        **kwargs
    ):
        #  dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        assert rectify == False
        # assert dropblock_prob == 0.0

        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        # self.dropblock_prob = dropblock_prob
        # if self.rectify:
        #     # from rfconv import RFConv2d
        #     self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
        #                          groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        # else:
        #     self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
        #                        groups=groups*radix, bias=bias, **kwargs)
        self.conv = Conv2d(
            in_channels,
            channels * radix,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
            **kwargs
        )
        # self.use_bn = norm is not None
        # if self.use_bn:
        #     self.bn0 = get_norm(norm, channels*radix)
        self.bn0 = get_norm(norm, channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        # if self.use_bn:
        #     self.bn1 = get_norm(norm, inter_channels)
        self.bn1 = get_norm(norm, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        # if dropblock_prob > 0.0:
        #     self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        # if self.use_bn:
        #     x = self.bn0(x)
        x = self.bn0(x)
        # if self.dropblock_prob > 0.0:
        #     x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        # if self.use_bn:
        #     gap = self.bn1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
