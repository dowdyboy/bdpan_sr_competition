import os
import random
import paddle
import paddle.vision.transforms.functional as F
import paddle.vision.transforms as T
from paddle.io import Dataset

import numpy as np
from PIL import Image
import cv2


def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class SRPairedRandomCrop(T.BaseTransform):
    """Super resolution random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        scale (int): model upscale factor.
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, patch_size, keys=None):
        self.patch_size = patch_size
        self.keys = keys

    def __call__(self, inputs):
        """inputs must be (lq_img or list[lq_img], gt_img or list[gt_img])"""
        x_patch_size = self.patch_size
        x2_patch_size = self.patch_size * 2
        x4_patch_size = self.patch_size * 4

        in_x = inputs[0]
        in_x2 = inputs[1]
        in_x4 = inputs[2]

        if isinstance(in_x, list):
            h_in_x, w_in_x, _ = in_x[0].shape
            h_in_x2, w_in_x2, _ = in_x2[0].shape
            h_in_x4, w_in_x4, _ = in_x4[0].shape
        else:
            h_in_x, w_in_x, _ = in_x.shape
            h_in_x2, w_in_x2, _ = in_x2.shape
            h_in_x4, w_in_x4, _ = in_x4.shape

        if h_in_x2 != h_in_x * 2 or w_in_x2 != w_in_x * 2 or \
            h_in_x4 != h_in_x * 4 or w_in_x4 != w_in_x * 4:
            raise ValueError('scale size not match')
        if h_in_x < x_patch_size or w_in_x < x_patch_size:
            raise ValueError('lq size error')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_in_x - x_patch_size)
        left = random.randint(0, w_in_x - x_patch_size)

        if isinstance(in_x, list):
            in_x = [
                v[top:top + x_patch_size, left:left + x_patch_size, ...]
                for v in in_x
            ]
            top_x2, left_x2 = int(top * 2), int(left * 2)
            in_x2 = [
                v[top_x2:top_x2 + x2_patch_size,
                left_x2:left_x2 + x2_patch_size, ...] for v in in_x2
            ]
            top_x4, left_x4 = int(top * 4), int(left * 4)
            in_x4 = [
                v[top_x4:top_x4 + x4_patch_size,
                left_x4:left_x4 + x4_patch_size, ...] for v in in_x4
            ]
        else:
            # crop lq patch
            in_x = in_x[top:top + x_patch_size, left:left + x_patch_size, ...]
            # crop corresponding gt patch
            top_x2, left_x2 = int(top * 2), int(left * 2)
            in_x2 = in_x2[top_x2:top_x2 + x2_patch_size,
                    left_x2:left_x2 + x2_patch_size, ...]
            top_x4, left_x4 = int(top * 4), int(left * 4)
            in_x4 = in_x4[top_x4:top_x4 + x4_patch_size,
                    left_x4:left_x4 + x4_patch_size, ...]

        outputs = (in_x, in_x2, in_x4)
        return outputs


class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):

    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            if isinstance(image, list):
                image = [F.hflip(v) for v in image]
            else:
                return F.hflip(image)
        return image


class PairedRandomVerticalFlip(T.RandomHorizontalFlip):

    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            if isinstance(image, list):
                image = [F.vflip(v) for v in image]
            else:
                return F.vflip(image)
        return image


class PairedRandomTransposeHW(T.BaseTransform):
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        prob (float): The propability to transpose the images.
        keys (list[str]): The images to be transposed.
    """

    def __init__(self, prob=0.5, keys=None):
        self.keys = keys
        self.prob = prob

    def _get_params(self, inputs):
        params = {}
        params['transpose'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['transpose']:
            if isinstance(image, list):
                image = [v.transpose(1, 0, 2) for v in image]
            else:
                image = image.transpose(1, 0, 2)
        return image


class SRDataset(Dataset):

    def __init__(self,
                 root_dir,
                 patch_size,
                 hflip_p=0.5,
                 vflip_p=0.5,
                 transpose_p=0.5,
                 use_normal=True,
                 use_cache=False,
                 is_val=False,
                 ):
        super(SRDataset, self).__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.use_normal = use_normal
        self.use_cache = use_cache
        self.is_val = is_val
        self.filepath_list = self._get_filepath_list()
        self.cache = {}
        self.crop_patch = SRPairedRandomCrop(patch_size, keys=['image', 'image', 'image'])
        self.hflip = PairedRandomHorizontalFlip(prob=hflip_p, keys=['image', 'image', 'image'])
        self.vflip = PairedRandomVerticalFlip(prob=vflip_p, keys=['image', 'image', 'image'])
        self.hw_transpose = PairedRandomTransposeHW(prob=transpose_p, keys=['image', 'image', 'image'])
        self.channel_transpose = T.Transpose(keys=['image', 'image', 'image'])
        self.to_tensor = T.ToTensor(keys=['image', 'image', 'image'])
        self.normalize = T.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.], keys=['image', 'image', 'image'])
        self.un_normalize = T.Normalize(mean=[0., 0., 0.], std=[1 / 255., 1 / 255., 1 / 255.], keys=['image', 'image', 'image'])

    def _get_filepath_list(self):
        ret = []
        for sub_dir in os.listdir(self.root_dir):
            sub_dir_path = os.path.join(self.root_dir, sub_dir)
            x_dir_path = os.path.join(sub_dir_path, 'x')
            x2_dir_path = os.path.join(sub_dir_path, 'x2')
            x4_dir_path = os.path.join(sub_dir_path, 'x4')
            for filename in os.listdir(x_dir_path):
                x_path = os.path.join(x_dir_path, filename)
                x2_path = os.path.join(x2_dir_path, filename)
                x4_path = os.path.join(x4_dir_path, filename)
                w, h = Image.open(x_path).size
                if min(w, h) > self.patch_size:
                    ret.append((x_path, x2_path, x4_path))
        return ret

    def __getitem__(self, idx):
        x_path, x2_path, x4_path = self.filepath_list[idx]
        if self.use_cache:
            if x_path in self.cache.keys():
                imgs = self.cache[x_path]
            else:
                imgs = (
                    load_image(x_path),
                    load_image(x2_path),
                    load_image(x4_path)
                )
                self.cache[x_path] = imgs
        else:
            imgs = (
                load_image(x_path),
                load_image(x2_path),
                load_image(x4_path)
            )
        if not self.is_val:
            imgs = self.crop_patch(imgs)
            imgs = self.hflip(imgs)
            imgs = self.vflip(imgs)
            imgs = self.hw_transpose(imgs)
        if self.use_normal:
            imgs = self.channel_transpose(imgs)
            imgs = self.normalize(imgs)
            imgs = [paddle.to_tensor(im) for im in imgs]
        else:
            imgs = self.to_tensor(imgs)
        return imgs[0], imgs[1], imgs[2]

    def __len__(self):
        return len(self.filepath_list)


class SRTestDataset(Dataset):
    
    def __init__(self, data_dir, use_normal=True, ):
        super(SRTestDataset, self).__init__()
        self.data_dir = data_dir
        self.use_normal = use_normal
        self.filepath_list = list(map(lambda x: os.path.join(self.data_dir, x), os.listdir(self.data_dir)))
        self.channel_transpose = T.Transpose()
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.], )
        self.un_normalize = T.Normalize(mean=[0., 0., 0.], std=[1 / 255., 1 / 255., 1 / 255.], )

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        im = load_image(filepath)
        if self.use_normal:
            im = self.channel_transpose(im)
            im = self.normalize(im)
            im = paddle.to_tensor(im)
        else:
            im = self.to_tensor(im)
        return im, filepath

    def __len__(self):
        return len(self.filepath_list)
