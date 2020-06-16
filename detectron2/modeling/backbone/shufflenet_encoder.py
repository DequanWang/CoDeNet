#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os.path as osp
import torch.nn as nn

from .shufflenet import ShuffleNetV2

_shufflenet_filename_mapper = {
    "V2_0.5x": "ShuffleNetV2.0.5x.pth.tar",
    "V2_1.0x": "ShuffleNetV2.1.0x.pth.tar",
    "V2_1.5x": "ShuffleNetV2.1.5x.pth.tar",
    "V2_2.0x": "ShuffleNetV2.2.0x.pth.tar",
    "V2+L": "ShuffleNetV2+.Large.pth.tar",
    "V2+M": "ShuffleNetV2+.Medium.pth.tar",
    "V2+S": "ShuffleNetV2+.Small.pth.tar",
}


_shufflenet_channels_mapper = {
    "V2_0.5x": [-1, 24, 48, 96, 192, 1024],
    "V2_1.0x": [-1, 24, 116, 232, 464, 1024],
    "V2_1.5x": [-1, 24, 176, 352, 704, 1024],
    "V2_2.0x": [-1, 24, 244, 488, 976, 2048],
    "V2+L": [-1, 16, 68, 168, 336, 672, 1280],
    "V2+M": [-1, 16, 48, 128, 256, 512, 1280],
    "V2+S": [-1, 16, 36, 104, 208, 416, 1280],
}


def build_shufflenet_encoder(cfg):
    arch_name = cfg.ENCODER.NAME
    base_folder = osp.split(osp.realpath(__file__))[0] + "/../../../pretrained/shufflenet/"
    arch = ShuffleNetV2(cfg.ENCODER.NORM, _shufflenet_channels_mapper[arch_name])
    arch.init_weights(filename=base_folder + _shufflenet_filename_mapper[arch_name])
    return arch
