# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transformer.py
import inspect
import numpy as np
import pprint
import sys
from abc import ABCMeta, abstractmethod
import cv2
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from PIL import Image

from .transform import ExtentTransform, ResizeTransform, RotationTransform

__all__ = [
    "CenterAffine",
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomCropPad",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomDistortion",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeShortestEdge",
    "TransformGen",
    "apply_transform_gens",
]


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[TransformGen] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


class TransformGen(metaclass=ABCMeta):
    """
    TransformGen takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


class RandomApply(TransformGen):
    """
    Randomly apply the wrapper transformation with a given probability.
    """

    def __init__(self, transform, prob=0.5):
        """
        Args:
            transform (Transform, TransformGen): the transform to be wrapped
                by the `RandomApply`. The `transform` can either be a
                `Transform` or `TransformGen` instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        assert isinstance(transform, (Transform, TransformGen)), (
            f"The given transform must either be a Transform or TransformGen instance. "
            f"Not {type(transform)}"
        )
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob
        self.transform = transform

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            if isinstance(self.transform, TransformGen):
                return self.transform.get_transform(img)
            else:
                return self.transform
        else:
            return NoOpTransform()


class RandomFlip(TransformGen):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class AffineTransform(Transform):
    """
    Augmentation from CenterNet
    """

    def __init__(self, src, dst, output_size):
        """
        output_size:(w, h)
        """
        super().__init__()
        affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img, self.affine, self.output_size, flags=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords


class PadTransform(Transform):
    def __init__(self, dw: list, dh: list, img_value=None, seg_value=None):
        super().__init__()
        assert img_value is not None or seg_value is not None
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Pad the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: padded image(s).
        """
        num_axis = len(img.shape)
        npad = [[0, 0] for x in range(num_axis)]
        if num_axis <= 3:
            npad[:2] = [self.dh, self.dw]
        else:
            npad[-3:-2] = [self.dh, self.dw]

        if self.img_value == "reflect":
            return np.pad(img, npad, mode="reflect")
        elif self.img_value == "symmetric":
            return np.pad(img, npad, mode="symmetric")
        else:
            return np.pad(img, npad, mode="constant", constant_values=self.img_value)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: padded coordinates.
        """
        coords[:, 0] += self.dw[0]
        coords[:, 1] += self.dh[0]
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply pad transform on a list of polygons, each represented by a Nx2 array.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: padded polygons.
        """
        return [self.apply_coords(p) for p in polygons]

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on the full-image segmentation.
        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: padded segmentation.
        """
        npad = [self.dh, self.dw]
        segmentation = np.pad(segmentation, npad, mode="constant", constant_values=self.seg_value)
        return segmentation


class CropPadTransform(Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        new_w: int,
        new_h: int,
        img_value=None,
        seg_value=None,
    ):
        super().__init__()
        self._set_attributes(locals())
        self.crop_trans = CropTransform(x0, y0, w, h)
        pad_width = (self.get_pad_width(w, new_w), self.get_pad_width(h, new_h))
        self.pad_trans = PadTransform(*pad_width, img_value, seg_value)

    def get_pad_width(self, ori: int, tar: int):
        pad_length = max(tar - ori, 0)
        pad_width = [pad_length // 2, pad_length // 2 + pad_length % 2]
        return pad_width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop and Pad the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped and padded image(s).
        """
        img = self.crop_trans.apply_image(img)
        img = self.pad_trans.apply_image(img)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped and padded coordinates.
        """
        coords = self.crop_trans.apply_coords(coords)
        coords = self.pad_trans.apply_coords(coords)
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop and pad transform on a list of polygons, each represented by a Nx2 array.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped and padded polygons.
        """
        polygons = self.crop_trans.apply_polygons(polygons)
        polygons = self.pad_trans.apply_polygons(polygons)
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on the full-image segmentation.
        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: cropped and padded segmentation.
        """
        segmentation = self.crop_trans.apply_segmentation(segmentation)
        segmentation = self.pad_trans.apply_segmentation(segmentation)
        return segmentation


class RandomCropPad(TransformGen):
    """
    Randomly crop and pad a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size, img_value=None, seg_value=None):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        h0 = np.random.randint(h - croph + 1) if h >= croph else 0
        w0 = np.random.randint(w - cropw + 1) if w >= cropw else 0
        dh = min(h, croph)
        dw = min(w, cropw)
        return CropPadTransform(w0, h0, dw, dh, cropw, croph, self.img_value, self.seg_value)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class CenterAffine(TransformGen):
    """
    Affine Transform for CenterNet
    """

    def __init__(self, boarder, output_size, random_aug=True):
        """
        Args:
            boarder(int): boarder size of image
            output_size(tuple): a tuple represents (width, height) of image
            random_aug(bool): whether apply random augmentation on annos or not
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        """
        generate one `AffineTransform` for input image
        """
        img_shape = img.shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        return AffineTransform(src, dst, self.output_size)

    @staticmethod
    def _get_boarder(boarder, size):
        """
        decide the boarder size of image
        """
        # NOTE This func may be reimplemented in the future
        i = 1
        size //= 2
        while size <= boarder // i:
            i *= 2
        return boarder // i

    def generate_center_and_scale(self, img_shape):
        r"""
        generate center and scale for image randomly
        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
            h_boarder = self._get_boarder(self.boarder, height)
            w_boarder = self._get_boarder(self.boarder, width)
            center[0] = np.random.randint(low=w_boarder, high=width - w_boarder)
            center[1] = np.random.randint(low=h_boarder, high=height - h_boarder)
        else:
            raise NotImplementedError("Non-random augmentation not implemented")

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


class DistortTransform(Transform):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the distorted image(s).
        """

        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = rand_scale(self.saturation)
        dexp = rand_scale(self.exposure)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img[:, :, 1] *= dsat
        img[:, :, 2] *= dexp
        H = img[:, :, 0] + dhue

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        img[:, :, 0] = H
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.asarray(img, dtype=np.float32)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class RandomDistortion(TransformGen):
    """
    Random distort image's hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        """
        RandomDistortion Initialization.
        Args:
            hue (float): value of hue
            saturation (float): value of saturation
            exposure (float): value of exposure
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return DistortTransform(self.hue, self.saturation, self.exposure)


class Resize(TransformGen):
    """ Resize image to a target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomRotation(TransformGen):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomCrop(TransformGen):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomExtent(TransformGen):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(TransformGen):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min=0.6, intensity_max=1.4):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(TransformGen):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min=0.6, intensity_max=1.4):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(TransformGen):
    """
    Randomly transforms image saturation.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min=0.6, intensity_max=1.4):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(TransformGen):
    """
    Randomly transforms image color using fixed PCA over ImageNet.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale=0.1):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )


def apply_transform_gens(transform_gens, img):
    """
    Apply a list of :class:`TransformGen` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens (list): list of :class:`TransformGen` instance to
            be applied.
        img (ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img)
        assert isinstance(
            tfm, Transform
        ), "TransformGen {} must return an instance of Transform! Got {} instead".format(g, tfm)
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)
