# CoDeNet

> [**CoDeNet: Algorithm-hardware Co-design for Deformable Convolution**](http://arxiv.org/abs/2006.08357)           
> Zhen Dong*, Dequan Wang*, Qijing Huang*, Yizhao Gao, Yaohui Cai, Bichen Wu, Kurt Keutzer, John Wawrzynek

The offical PyTorch implementation of [CoDeNet](docs/CoDeNet.pdf), based on [detectron2](https://github.com/facebookresearch/detectron2/).


## Abstract

Deploying deep learning models on embedded systems for computer vision tasks has been challenging due to limited compute resources and strict energy budgets. The majority of existing work focuses on accelerating image classification, while other fundamental vision problems, such as object detection, have not been adequately addressed. Compared with image classification, detection problems are more sensitive to the spatial variance of objects, and therefore, require specialized convolutions to aggregate spatial information. To address this, recent work proposes dynamic deformable convolution to augment regular convolutions. Regular convolutions process a fixed grid of pixels across all the spatial locations in an image, while dynamic deformable convolution may access arbitrary pixels in the image and the access pattern is inputdependent and varies per spatial location. These properties lead to inefficient memory accesses of inputs with existing hardware. In this work, we first investigate the overhead of the deformable convolution on embedded FPGA SoCs, and introduce a depthwise deformable convolution to reduce the total number of operations required. We then show the speed-accuracy tradeoffs for a set of algorithm modifications including irregular-access versus limitedrange and fixed-shape. We evaluate these algorithmic changes with corresponding hardware optimizations. Results show a 1.36× and 9.76× speedup respectively for the full and depthwise deformable convolution on the embedded FPGA accelerator with minor accuracy loss on the object detection task. We then co-design an efficient network CoDeNet with the modified deformable convolution for object detection and quantize the network to 4-bit weights and 8-bit activations. Results show that our designs lie on the pareto-optimal front of the latency-accuracy tradeoff for the object detection task on embedded FPGAs.


## Installation

See [INSTALL.md](INSTALL.md) and [GETTING_STARTED.md](GETTING_STARTED.md).
Learn more at detectron2's [documentation](https://detectron2.readthedocs.org).


## Experiments

> **_NOTE:_** We use the pre-trained [ShuffleNet V2 1.0x](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2) as the default backbone.

### Main Results

```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_square_depthwise.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_square_depthwise
# result: AP: 41.7	AP50: 64.5	AP75: 43.8
```


```bash
python tools/train_net.py --num-gpus 10 --config-file configs/centernet/coco/V2_1.0x_coco_512_10gpus_1x_deform_conv_square_depthwise.yaml
# folder: output/centernet/coco/V2_1.0x_coco_512_10gpus_1x_deform_conv_square_depthwise
# result: AP: 21.6	AP50: 37.4	AP75: 21.8	APs: 6.5	APm: 23.7	APl: 34.8
```

### Ablation Study

```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_3x3.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_3x3
# result: AP: 37.9	AP50: 61.0	AP75: 39.9
```

```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_3x3_depthwise.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_3x3_depthwise
# result: AP: 37.0	AP50: 60.8	AP75: 38.3
```

```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_original.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_original
# result: AP: 45.1	AP50: 67.8	AP75: 47.7
```


```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_bound.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_bound
# result: AP: 44.3	AP50: 66.5	AP75: 47.1
```


```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_square.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_square
# result: AP: 42.2	AP50: 64.9	AP75: 45.1
```


```bash
python tools/train_net.py --num-gpus 4 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_original_depthwise.yaml
# folder: output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_original_depthwise
# result: AP: 43.9	AP50: 66.5	AP75: 46.5
```


## Acknowledgement

- [Detectron2](https://github.com/facebookresearch/Detectron2)
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better](https://github.com/FateScript/CenterNet-better)
- [ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series)


## Citation

```BibTeX
@article{dong2020codenet,
  title={CoDeNet: Algorithm-hardware Co-design for Deformable Convolution},
  author={Dong, Zhen, and Wang, Dequan and Huang, Qijing and Gao, Yizhao and Cai, Yaohui and Wu, Bichen and Keutzer, Kurt and Wawrzynek, John},
  journal={arXiv preprint arXiv:2006.08357},
  year={2020}
}
```
