import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

from detectron2.data.transforms.transform_gen import CenterAffine
from detectron2.modeling.backbone import build_shufflenet_decoder, build_shufflenet_encoder
from detectron2.structures import Boxes, ImageList, Instances

from .build import META_ARCH_REGISTRY


# https://github.com/FateScript/CenterNet-better/blob/master/dl_lib/nn_utils/feature_utils.py
def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def modified_focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 2, gamma: float = 4
) -> torch.Tensor:
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, gamma)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.sigmoid(inputs)
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum().float()
    if num_pos == 0:
        loss = -neg_loss.sum()
    else:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


cls_loss = torch.jit.script(modified_focal_loss)  # type: torch.jit.ScriptModule


def reg_loss(output, mask, index, target, loss_type="l1", smooth_l1_beta=0.1):
    pred = gather_feature(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    if loss_type == "l1":
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    elif loss_type == "smooth_l1":
        loss = smooth_l1_loss(pred * mask, target * mask, smooth_l1_beta, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


# https://github.com/princeton-vl/CornerNet/blob/3e71377b45098f9cea26d5a39de0138174c90d49/sample/utils.py
def get_gaussian_radius(box_size, min_overlap):
    """
    copyed from CornerNet
    box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
    notice: we are using the fixed version in CornerNet
    """
    box_tensor = torch.Tensor(box_size)
    width, height = box_tensor[..., 0], box_tensor[..., 1]

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return torch.min(r1, torch.min(r2, r3))


def draw_gaussian(fmap, center, radius, k=1):
    sigma = (2 * radius + 1) / 6
    m, n = radius, radius
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    gaussian = torch.Tensor(gaussian)
    x, y = int(center[0]), int(center[1])
    height, width = fmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_fmap = fmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
        masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
        fmap[y - top : y + bottom, x - left : x + right] = masked_fmap


def get_topk_from_scores(scores, K):
    """
    get top K point in score map
    """
    batch, channel, height, width = scores.shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    # div by K because index is grouped by K(C x K shape)
    topk_clses = (index / K).int()
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_pred_dict(pred_dict, K=100, pseudo_nms_pool_size=3, cat_spec_wh=False):
    r"""
    decode output feature map to detection results
    Args:
        fmap(Tensor): output feature map
        wh(Tensor): tensor that represents predicted width-height
        reg(Tensor): tensor that represens regression of center points
        cat_spec_wh(bool): whether apply gather on tensor `wh` or not
        K(int): topk value
    """
    fmap = torch.sigmoid(pred_dict["cls"])
    reg, wh = pred_dict["reg"], pred_dict["wh"]
    batch, channel, height, width = fmap.shape

    # fmap = CenterNetDecoder.pseudo_nms(fmap)

    """
    apply max pooling to get the same effect of nms

        fmap(Tensor): output tensor of previous step
        pool_size(int): size of max-pooling
    """

    fmap_max = F.max_pool2d(
        fmap, pseudo_nms_pool_size, stride=1, padding=(pseudo_nms_pool_size - 1) // 2
    )
    scores = fmap * (fmap_max == fmap).float()

    scores, index, clses, ys, xs = get_topk_from_scores(scores, K)

    if reg is not None:
        reg = gather_feature(reg, index, use_transform=True)
        reg = reg.reshape(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = gather_feature(wh, index, use_transform=True)

    if cat_spec_wh:
        wh = wh.view(batch, K, channel, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).reshape(batch, K, 2)
    else:
        wh = wh.reshape(batch, K, 2)

    clses = clses.reshape(batch, K, 1).float()
    scores = scores.reshape(batch, K, 1)

    half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
    bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

    scores = scores.reshape(-1)
    clses = clses.reshape(-1).to(torch.int64)

    return (bboxes, scores, clses)


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.test_topk        = cfg.TEST.DETECTIONS_PER_IMAGE
        self.num_det_max      = cfg.MODEL.CENTERNET.NUM_DET_MAX
        self.num_classes      = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.down_scale       = cfg.MODEL.CENTERNET.DOWN_SCALE
        self.output_size      = cfg.MODEL.CENTERNET.OUTPUT_SIZE
        self.min_overlap      = cfg.MODEL.CENTERNET.MIN_OVERLAP
        self.loss_cls_weight  = cfg.MODEL.CENTERNET.LOSS.CLS_WEIGHT
        self.loss_wh_weight   = cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT
        self.loss_reg_weight  = cfg.MODEL.CENTERNET.LOSS.REG_WEIGHT
        self.wh_loss_type     = cfg.MODEL.CENTERNET.LOSS.WH_LOSS_TYPE
        self.reg_loss_type    = cfg.MODEL.CENTERNET.LOSS.REG_LOSS_TYPE
        self.focal_loss_alpha = cfg.MODEL.CENTERNET.LOSS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.CENTERNET.LOSS.FOCAL_LOSS_GAMMA
        # fmt: on

        self.encoder = build_shufflenet_encoder(cfg.MODEL.CENTERNET)
        self.decoder, self.head = build_shufflenet_decoder(cfg.MODEL.CENTERNET)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            return self.inference(images)
        else:
            feat_enc = self.encoder(images.tensor)
            feat_dec = self.decoder(feat_enc)
            pred_dict = self.head(feat_dec)
            gt_dict = self.get_ground_truth(batched_inputs)
            return self.losses(pred_dict, gt_dict)

    def losses(self, pred_dict, gt_dict):
        r"""
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "score_map": gt scoremap,
                "wh": gt width and height of boxes,
                "reg": gt regression of box center point,
                "reg_mask": mask of regression,
                "index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
            "cls": predicted score map
            "reg": predcited regression
            "wh": predicted width and height of box
        }
        """
        # scoremap loss
        pred_score = pred_dict["cls"]
        cur_device = pred_score.device
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(cur_device)

        # loss_cls = modified_focal_loss(pred_score, gt_dict['score_map'])
        loss_cls = cls_loss(
            pred_score, gt_dict["score_map"], self.focal_loss_alpha, self.focal_loss_gamma
        )

        mask = gt_dict["reg_mask"]
        index = gt_dict["index"]
        index = index.to(torch.long)

        # width and height loss, better version
        loss_wh = reg_loss(pred_dict["wh"], mask, index, gt_dict["wh"], self.wh_loss_type)

        # regression loss
        loss_reg = reg_loss(pred_dict["reg"], mask, index, gt_dict["reg"], self.reg_loss_type)

        loss_cls *= self.loss_cls_weight
        loss_wh *= self.loss_wh_weight
        loss_reg *= self.loss_reg_weight

        loss = {"loss_cls": loss_cls, "loss_wh": loss_wh, "loss_reg": loss_reg}
        # print(loss)
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        scoremap_list, wh_list, reg_list, reg_mask_list, index_list = [[] for i in range(5)]

        for data in batched_inputs:
            bbox_dict = data["instances"].get_fields()
            boxes, classes = bbox_dict["gt_boxes"], bbox_dict["gt_classes"]
            num_boxes = boxes.tensor.shape[0]
            assert num_boxes <= self.num_det_max
            boxes.scale(1 / self.down_scale, 1 / self.down_scale)

            # init gt tensors
            gt_scoremap = torch.zeros(self.num_classes, *self.output_size)
            gt_wh = torch.zeros(self.num_det_max, 2)
            gt_reg = torch.zeros_like(gt_wh)
            reg_mask = torch.zeros(self.num_det_max)
            gt_index = torch.zeros(self.num_det_max)

            centers = boxes.get_centers()
            centers_int = centers.to(torch.int32)
            gt_index[:num_boxes] = centers_int[..., 1] * self.output_size[0] + centers_int[..., 0]
            gt_reg[:num_boxes] = centers - centers_int
            reg_mask[:num_boxes] = 1

            wh, box_tensor = torch.zeros_like(centers), boxes.tensor
            wh[..., 0] = box_tensor[..., 2] - box_tensor[..., 0]
            wh[..., 1] = box_tensor[..., 3] - box_tensor[..., 1]

            radius = get_gaussian_radius(wh, self.min_overlap)
            radius = torch.clamp_min(radius, 0)
            radius = radius.type(torch.int).cpu().numpy()
            for i in range(classes.shape[0]):
                channel_index = classes[i]
                draw_gaussian(gt_scoremap[channel_index], centers_int[i], radius[i])

            gt_wh[:num_boxes] = wh

            scoremap_list.append(gt_scoremap)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(reg_mask)
            index_list.append(gt_index)

        gt_dict = {
            "score_map": torch.stack(scoremap_list, dim=0),
            "wh": torch.stack(wh_list, dim=0),
            "reg": torch.stack(reg_list, dim=0),
            "reg_mask": torch.stack(reg_mask_list, dim=0),
            "index": torch.stack(index_list, dim=0),
        }
        return gt_dict

    @torch.no_grad()
    def inference(self, images):
        """
        image(tensor): ImageList in dl_lib.structures
        """
        n, c, h, w = images.tensor.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)

        img_info = dict(
            center=center_wh,
            size=size_wh,
            height=new_h // self.down_scale,
            width=new_w // self.down_scale,
        )

        pad_value = [-x / y for x, y in zip(self.pixel_mean, self.pixel_std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(images.tensor.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h : h + pad_h, pad_w : w + pad_w] = images.tensor

        feat_enc = self.encoder(aligned_img)
        feat_dec = self.decoder(feat_enc)
        pred_dict = self.head(feat_dec)

        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        """
        bboxes, scores, classes = decode_pred_dict(pred_dict, self.test_topk)
        """
        transform predicted boxes to target boxes

        bboxes(Tensor): torch Tensor with (Batch, N, 4) shape
        img_info(dict): dict contains all information of original image
        """
        boxes = bboxes.cpu().numpy().reshape(-1, 4)
        center = img_info["center"]
        size = img_info["size"]
        output_size = (img_info["width"], img_info["height"])
        src, dst = CenterAffine.generate_src_and_dst(center, size, output_size)
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        coords = boxes.reshape(-1, 2)
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        bboxes = Boxes(np.dot(aug_coords, trans.T).reshape(-1, 4))

        ori_w, ori_h = img_info["center"] * 2
        det_instance = Instances(
            (int(ori_h), int(ori_w)), pred_boxes=bboxes, scores=scores, pred_classes=classes
        )

        return [{"instances": det_instance}]

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.encoder.size_divisibility)
        return images
