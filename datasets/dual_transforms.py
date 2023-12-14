import numpy as np
import random
import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any
import torch
from torch import nn
from albumentations import DualTransform
import torch.nn.functional as F
import os
import PIL.Image as Image
import torchvision.transforms as T

def cutout(
    img, per_pixel, rand_color, holes: Iterable[Tuple[int, int, int, int]], 
    fill_value: Union[int, float] = 0, dtype=torch.float32):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.clone()
    chan = img.size(0)
    for x1, y1, x2, y2 in holes:
        patch_size = (chan, y2-y1, x2-x1)
        if per_pixel:
            img[:, y1:y2, x1:x2] = torch.empty(patch_size, dtype=dtype).normal_()
        elif rand_color:
            img[:, y1:y2, x1:x2] = torch.empty((chan, 1, 1), dtype=dtype).normal_()
        else:
            img[:, y1:y2, x1:x2] = fill_value
    return img

class DualRE(DualTransform):

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1/3,
        min_aspect=0.3,
        max_aspect=None,
        mode='const',
        min_count=1,
        max_count=None,
        # device='cuda',
        always_apply: bool = False,
        fill_value=0,
        mask_fill_value=0,
    ):
        super(DualRE, self).__init__(always_apply, probability)
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

    def apply(
        self,
        img,
        per_pixel=True,
        rand_color=False,
        fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ):
        return cutout(img, per_pixel, rand_color, holes, fill_value)

    def apply_to_mask(
        self,
        img,
        mask_fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ):
        if mask_fill_value is None:
            return img
        return cutout(img, False, False, holes, mask_fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[1:]

        area = height * width
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)

        holes = []
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < width and h < height:
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)
                    bottom = top + h
                    right = left + w
                    holes.append((left, top, right, bottom))
                    break
        return {"holes": holes}
    
    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        if hasattr(self, "rand_color"):
            params['rand_color'] = self.rand_color
        if hasattr(self, "per_pixel"):
            params['per_pixel'] = self.per_pixel
        params.update({"cols": kwargs["image"].shape[2], "rows": kwargs["image"].shape[1]})
        return params
    
    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "min_area",
            "max_area",
            "min_aspect",
            "max_aspect",
            "mode",
            "min_count",
            "max_count",
            "fill_value",
            "mask_fill_value",
            "rand_color",
            "per_pixel",
        )
    

class DualOcc(DualTransform):

    def __init__(
        self,
        probability=0.5,
        # device='cuda',
        always_apply: bool = False,
        mask_fill_value=0,
        root=None,
        normalizer=None,
    ):
        super(DualOcc, self).__init__(always_apply, probability)
        self.probability = probability
        self.mask_fill_value = mask_fill_value
        self.root = root
        self.occ_imgs = os.listdir(self.root)
        for img in self.occ_imgs:
            if not img.endswith('.jpg'):
                self.occ_imgs.remove(img)
        self.len = len(self.occ_imgs)
        self.ratio_thre = 2
        self.normalizer = normalizer

    def apply(
        self,
        img,
        occ_img,
        hole: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ):  
        y1, x1, y2, x2 = hole
        img[:, y1:y2, x1:x2] = occ_img
        return img

    def apply_to_mask(
        self,
        img,
        mask_fill_value: Union[int, float] = 0,
        hole: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ):
        if mask_fill_value is None:
            return img
        y1, x1, y2, x2 = hole
        img[:, y1:y2, x1:x2] = mask_fill_value
        return img

    def get_params_dependent_on_targets(self, params):

        img = params["image"]
        h, w = img.shape[1:]

        index = random.randint(0, self.len-1)

        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')
        h_, w_ = occ_img.height, occ_img.width
        ratio = h_ / w_

        if ratio > self.ratio_thre:
            # re_size = (random.randint(h//2, h), random.randint(w//4, w//2))
            re_size = (h, random.randint(w//4, w//2))
            function = T.Compose([
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                self.normalizer
            ])
            occ_img = function(occ_img)
        else:
            # re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
            re_size = (random.randint(h//4, h//2), w)
            function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                self.normalizer
            ])
            occ_img = function(occ_img)

        h_, w_ = re_size[0], re_size[1]
        index_ = random.randint(0, 3)

        if index_==0:
            hole = [0, 0, h_, w_]
        elif index_==1:
            hole = [0, w-w_, h_, w]
        elif index_==2:
            hole = [h-h_, 0, h, w_]
        else:
            hole = [h-h_, w-w_, h, w]

        return {"hole": hole,
                "occ_img": occ_img,
                }
    
    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        params.update({"cols": kwargs["image"].shape[2], "rows": kwargs["image"].shape[1]})
        return params
    
    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "mask_fill_value",
            "root",
            "occ_imgs",
            "len",
            "ratio_thre",
            "normalizer",
        )