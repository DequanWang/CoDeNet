import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from detectron2.layers import get_norm


# modified from https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/blocks.py
class ShuffleV2Block(nn.Module):
    def __init__(self, norm_type, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            get_norm(norm_type, mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False
            ),
            get_norm(norm_type, mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            get_norm(norm_type, outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                get_norm(norm_type, inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                get_norm(norm_type, inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


# modified from https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/network.py
class ShuffleNetV2(nn.Module):
    def __init__(self, norm_type, channels):
        super(ShuffleNetV2, self).__init__()
        self.size_divisibility = 32
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            get_norm(norm_type, input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(
                            norm_type,
                            input_channel,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=2,
                        )
                    )
                else:
                    self.features.append(
                        ShuffleV2Block(
                            norm_type,
                            input_channel // 2,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=1,
                        )
                    )

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            get_norm(norm_type, self.stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        return x

    def init_weights(self, filename=None, map_location="cpu", strict=False):
        if isinstance(filename, str):
            # load_checkpoint(self, filename, map_location, strict)
            # modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/checkpoint.py
            """Load checkpoint from a file or URI.
            Args:
                model (Module): Module to load checkpoint.
                filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
                map_location (str): Same as :func:`torch.load`.
                strict (bool): Whether to allow different params for the model and
                    checkpoint.
            Returns:
                dict or OrderedDict: The loaded checkpoint.
            """
            checkpoint = torch.load(filename, map_location=map_location)
            # get state_dict from checkpoint
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            # load state_dict
            self.load_state_dict(state_dict, strict)
            # return checkpoint
        else:
            raise TypeError("filename must be a str")
