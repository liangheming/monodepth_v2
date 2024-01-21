import os
import torch
import random
import numbers
import cv2 as cv
import numpy as np

from copy import deepcopy
from typing import List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomErasing
from torchvision.transforms import functional as f


class MultiRandomErasing(RandomErasing):
    def __init__(self, img_id=1, start_times=10, end_times=20, scale=(0.001, 0.004), ratio=(0.5, 2.0), value=0,
                 inplace=False):
        super(MultiRandomErasing, self).__init__(p=1.0, scale=scale, ratio=ratio, value=value, inplace=inplace)
        self.img_id = img_id
        self.start_times = start_times
        self.end_times = end_times

    def forward(self, images):
        times = random.randint(self.start_times, self.end_times)
        for _ in range(times):
            images[self.img_id] = super(MultiRandomErasing, self).forward(images[self.img_id])
        return images


class SyncColorJitter(torch.nn.Module):
    def __init__(
            self,
            brightness: Union[float, Tuple[float, float]] = 0,
            contrast: Union[float, Tuple[float, float]] = 0,
            saturation: Union[float, Tuple[float, float]] = 0,
            hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    @staticmethod
    def get_params(
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, images):
        """
        Args:
            images (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                images = [f.adjust_brightness(img, brightness_factor) for img in images]
            elif fn_id == 1 and contrast_factor is not None:
                images = [f.adjust_contrast(img, contrast_factor) for img in images]
            elif fn_id == 2 and saturation_factor is not None:
                images = [f.adjust_saturation(img, saturation_factor) for img in images]
            elif fn_id == 3 and hue_factor is not None:
                images = [f.adjust_hue(img, hue_factor) for img in images]

        return images

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class Resize(torch.nn.Module):
    def __init__(self, height, width):
        super(Resize, self).__init__()
        self.height = height
        self.width = width

    def forward(self, images):
        return [f.resize(img, [self.height, self.width], f.InterpolationMode.LANCZOS) for img in images]


class HFlip(torch.nn.Module):
    def __init__(self):
        super(HFlip, self).__init__()

    def forward(self, images):
        return [f.hflip(image) for image in images]


class ToTensor(torch.nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, images):
        return [f.to_tensor(image) for image in images]


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class KittiDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split_file,
                 frame_ids,
                 height=192,
                 width=640,
                 train=True,
                 scales=range(4)):
        self.train = train
        self.data_dir = data_dir
        self.split_file = split_file
        self.height = height
        self.width = width
        self.scales = scales
        self.gt_data = self.check_gt()
        if self.gt_data is not None:
            frame_ids = [0]
        self.frame_ids = frame_ids
        self.data_list = self.get_samples()

        self.color_gitter = SyncColorJitter(brightness=(0.8, 1.2),
                                            contrast=(0.8, 1.2),
                                            saturation=(0.8, 1.2),
                                            hue=(-0.1, 0.1))
        self.resize = Resize(height=height, width=width)
        self.h_flip = HFlip()
        self.to_tensor = ToTensor()
        self.normalized_k = np.array([[0.58, 0, 0.5],
                                      [0, 1.92, 0.5],
                                      [0, 0, 1]], dtype=np.float32)
        self.k = self.normalized_k.copy()
        self.k[0, :] *= width
        self.k[1, :] *= height
        self.inv_k = np.linalg.pinv(self.k)

        self.inv_k[[0, 1, 2, 2], [1, 0, 0, 1]] = 0.0

    def check_gt(self):
        if self.train:
            return None
        gt_path = os.path.join(os.path.dirname(self.split_file), "gt_depths.npz")
        if os.path.exists(gt_path):
            return [cv.resize(d, (self.width, self.height), interpolation=cv.INTER_NEAREST) for d in
                    np.load(gt_path, allow_pickle=True)["data"]]
        return None

    def __getitem__(self, item):
        sample = self.data_list[item]
        img_list = list()
        ret_dict = dict()
        for frame_id in self.frame_ids:
            if frame_id == 's':
                frame_id = 'lr' if sample.get('lr', None) is not None else 'rl'
            img_list.append(pil_loader(sample[frame_id]))
        img_list = self.resize(img_list)
        is_flip = self.train and random.random() > 0.5

        if is_flip:
            img_list = self.h_flip(img_list)

        # multi level raw images
        raw_idx = self.frame_ids.index(0)
        for s in self.scales[1:]:
            scale_ratio = 2 ** s
            ret_dict[("raw", 0, s)] = f.to_tensor(
                f.resize(img_list[raw_idx],
                         size=[self.height // scale_ratio, self.width // scale_ratio],
                         interpolation=f.InterpolationMode.LANCZOS))

        aug_list = deepcopy(img_list)
        if self.train and random.random() > 0.0:
            aug_list = self.color_gitter(aug_list)
        img_list = self.to_tensor(img_list)
        aug_list = self.to_tensor(aug_list)

        for frame_id, raw, aug in zip(self.frame_ids, img_list, aug_list):
            ret_dict[('raw', frame_id, 0)] = raw
            ret_dict[('aug', frame_id, 0)] = aug
            ret_dict['k'] = torch.from_numpy(self.k.copy())
            ret_dict['inv_k'] = torch.from_numpy(self.inv_k.copy())
            if frame_id == "s":
                direction = 1.0 if sample.get('rl', None) is not None else -1.0
                flip_direction = -1.0 if is_flip else 1.0
                ret_dict['stereo_rot'] = torch.eye(3).float()
                stereo_trans = torch.zeros(3)
                stereo_trans[0] = (flip_direction * direction * 0.1)
                ret_dict['stereo_trans'] = stereo_trans

        if self.gt_data is not None:
            ret_dict['gt_data'] = torch.from_numpy(self.gt_data[item])
        return ret_dict

    def __len__(self):
        return len(self.data_list)

    def get_samples(self):
        ret = list()
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        rf = open(self.split_file, 'r')
        for line in rf.readlines():
            dir_name, frame_id, camera_id = line.split()
            frame_id = int(frame_id)
            ret_dict = dict()
            for frame in self.frame_ids:
                if frame == 's':
                    img_name = "{:0>10d}.jpg".format(frame_id)
                    other_id = 'l' if camera_id == 'r' else 'r'
                    mid_dir = "image_0{:d}/data".format(side_map[other_id])
                    key_name = 'lr' if camera_id == 'l' else 'rl'
                else:
                    key_name = frame
                    img_name = "{:0>10d}.jpg".format(frame + frame_id)
                    mid_dir = "image_0{:d}/data".format(side_map[camera_id])
                img_path = os.path.join(self.data_dir, dir_name, mid_dir, img_name)
                assert os.path.exists(img_path), "{:s} is not exit".format(img_path)
                ret_dict[key_name] = img_path
            ret.append(ret_dict)
        rf.close()
        return ret

    def collate_fn(self, batch):
        ret_dict = dict()
        for b in batch:
            for k, v in b.items():
                out = ret_dict.get(k, list())
                if len(out) == 0:
                    ret_dict[k] = out
                out.append(v)
            del b
        for k in ret_dict.keys():
            ret_dict[k] = torch.stack(ret_dict[k], dim=0)
        return ret_dict


def build_dataset(args, **kwargs):
    split_file_name = "train_files.txt" if kwargs['is_train'] else "val_files.txt"
    split_file = os.path.join("splits", args.split, split_file_name)
    dataset = KittiDataset(data_dir=args.kitti_dir,
                           split_file=split_file,
                           frame_ids=args.frame_ids,
                           height=args.height,
                           width=args.width,
                           scales=args.scales,
                           train=kwargs['is_train'])

    return dataset


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    data = KittiDataset(data_dir="/home/lion/large_data/data/kitti/raw",
                        split_file="../splits/eigen_zhou/val_files_bak.txt",
                        frame_ids=[-1, 0, 1, 's'],
                        train=True)
    dload = DataLoader(dataset=data, batch_size=4, shuffle=True, drop_last=True, num_workers=4,
                       collate_fn=data.collate_fn)

    for d in dload:
        print(d[('raw', -1, 0)].shape)
        break
