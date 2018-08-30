import torch
import torchvision
import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class CenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class RandomHorizontalFlip(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class RandomResizedCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            scale = Resize(self.size, interpolation=self.interpolation)
            return scale(img_group)


class Normalize(object):
    def __init__(self, mean, std, channel=3, num_frames=16, size=160):
        self.mean = mean
        self.std = std
        self.channel = channel
        self.num_frames = num_frames
        self.size = size

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[1])
        rep_std = self.std * (tensor.size()[1])
        tensor = tensor.contiguous().view(self.channel * self.num_frames, self.size, self.size)
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        tensor = tensor.view(self.channel, self.num_frames, self.size, self.size)
        return tensor


class ToTensor(object):
    def __init__(self):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        img_group = [img.numpy() for img in img_group]
        img_group = np.array(img_group)
        img_group = img_group.transpose(1, 0, 2, 3)
        img_group = torch.from_numpy(img_group)
        return img_group
