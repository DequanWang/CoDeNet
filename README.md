# CoDeNet

> [**Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs**](http://arxiv.org/abs/2006.08357)           
> Qijing Huang*, Dequan Wang*, Zhen Dong*, Yizhao Gao, Yaohui Cai, Bichen Wu, Tian Li, Kurt Keutzer, John Wawrzynek           
> The 2021 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays

The offical PyTorch implementation of [CoDeNet](docs/CoDeNet.pdf), based on [detectron2](https://github.com/facebookresearch/detectron2/). Please see [here](https://github.com/hqjenny/CoDeNet) for more details on hardware acceleration.


## Abstract

Deploying deep learning models on embedded systems for computer vision tasks has been challenging due to limited compute resources and strict energy budgets. The majority of existing work focuses on accelerating image classification, while other fundamental vision problems, such as object detection, have not been adequately addressed. Compared with image classification, detection problems are more sensitive to the spatial variance of objects, and therefore, require specialized convolutions to aggregate spatial information. To address this need, recent work introduces dynamic deformable convolution to augment regular convolutions. Regular convolutions process a fixed grid of pixels across all the spatial locations in an image, while dynamic deformable convolution may access arbitrary pixels in the image with the access pattern being input-dependent and varying with spatial location. These properties lead to inefficient memory accesses of inputs with existing hardware. 

In this work, we harness the flexibility of FPGAs to develop a novel object detection pipeline with deformable convolutions. We show the speed-accuracy tradeoffs for a set of algorithm modifications including irregular-access versus limited-range and fixed-shape on a flexible hardware accelerator. We evaluate these algorithmic changes with corresponding hardware optimizations and show a 1.36× and 9.76× speedup respectively for the full and depthwise deformable convolution on hardware with minor accuracy loss. We then **Co-De**sign a **Net**work **CoDeNet** with the modified deformable convolution for object detection and quantize the network to 4-bit weights and 8-bit activations. With our high-efficiency implementation, our solution reaches 26.9 frames per second with a tiny model size of 0.76 MB while achieving 61.7 AP50 on the standard object detection dataset, Pascal VOC. With our higher accuracy implementation, our model gets to 67.1 AP50 on Pascal VOC with only 2.9 MB of parameters—20.9× smaller but 10% more accurate than Tiny-YOLO.


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
@inproceedings{huang2021efficient,
  title={Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs},
  author={Huang, Qijing and Wang, Dequan and Dong, Zhen, and Gao, Yizhao and Cai, Yaohui and Wu, Bichen and Li, Tian and Keutzer, Kurt and Wawrzynek, John},
  booktitle={The 2021 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
  year={2021}
}
```

```BibTeX
@article{dong2020codenet,
  title={CoDeNet: Algorithm-hardware Co-design for Deformable Convolution},
  author={Dong, Zhen, and Wang, Dequan and Huang, Qijing and Gao, Yizhao and Cai, Yaohui and Wu, Bichen and Keutzer, Kurt and Wawrzynek, John},
  journal={arXiv preprint arXiv:2006.08357},
  year={2020}
}
```
