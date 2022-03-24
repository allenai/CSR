import random

import torchvision.transforms.functional as F
from src.shared.constants import (COLOR_JITTER_BRIGHTNESS,
                                  COLOR_JITTER_CONTRAST, COLOR_JITTER_HUE,
                                  COLOR_JITTER_SATURATION,
                                  GRAYSCALE_PROBABILITY, IMAGE_SIZE, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD)
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

class Compose(object):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            t(data)
        return data


class ToZeroOne(object):
    def __init__(self):
        super().__init__()
        self.toTensor = T.ToTensor()

    def __call__(self, data):
        if 'image' in data and type(data['image']) != torch.Tensor:
            data['image'] = self.toTensor(data['image'])
        if 'shuffle_image' in data:
            data['shuffle_image'] = self.toTensor(data['shuffle_image'])
        # if 'mask_1' in data:
        #     data['mask_1'] = self.toTensor(data['mask_1'])
        # if 'mask_2' in data:
        #     data['mask_2'] = self.toTensor(data['mask_2'])


class Normalize(object):
    """ImageNet RGB normalization."""

    def __init__(self, mean, std):
        super().__init__()
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, data):
        if 'image' in data:
            data['image'] = self.normalize(data['image'])
        if 'shuffle_image' in data:
            data['shuffle_image'] = self.normalize(data['shuffle_image'])


class ColorJitter(object):
    """[summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__()
        self.colorJitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data):
        """[summary]

        Args:
            curr_image ([type]): [description]
            next_image ([type]): [description]

        Returns:
            [type]: [description]
        """
        if 'image' in data:
            data['image'] = self.colorJitter(data['image'])
        if 'shuffle_image' in data:
            data['shuffle_image'] = self.colorJitter(data['shuffle_image'])


class RandomGrayscale(object):
    """[summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self, p):
        super().__init__()
        self.grayscale = T.RandomGrayscale(p=p)

    def __call__(self, data):
        if 'image' in data:
            data['image'] = self.grayscale(data['image'])
        if 'shuffle_image' in data:
            data['shuffle_image'] = self.grayscale(data['shuffle_image'])


class Rotate:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, data):
        angle = random.choice(self.angles)
        if 'image' in data:
            data['image'] = F.rotate(data['image'], angle)

        if 'mask_1' in data:
            data['mask_1'] = F.rotate(data['mask_1'], angle)

        if 'mask_2' in data:
            data['mask_2'] = F.rotate(data['mask_2'], angle)

        if 'shuffle_image' in data:
            data['shuffle_image'] = F.rotate(data['shuffle_image'], angle)

        if 'shuffle_mask_1' in data:
            data['shuffle_mask_1'] = F.rotate(data['shuffle_mask_1'], angle)

        if 'shuffle_mask_2' in data:
            data['shuffle_mask_2'] = F.rotate(data['shuffle_mask_2'], angle)


class Blur:
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        if 'image' in data:
            data['image'] = F.gaussian_blur(data['image'], 3)

class Resize:
    """Rotate by one of the given angles."""

    def __init__(self, size):
        self.size = size
        self.interp = InterpolationMode.BILINEAR

    def __call__(self, data):
        if 'image' in data:
            data['image'] = F.resize(data['image'], (self.size, self.size), interpolation=self.interp)

        if 'mask_1' in data:
            data['mask_1'] = F.resize(data['mask_1'], (self.size, self.size), interpolation=self.interp)

        if 'mask_2' in data:
            data['mask_2'] = F.resize(data['mask_2'], (self.size, self.size), interpolation=self.interp)

        if 'shuffle_image' in data:
            data['shuffle_image'] = F.resize(data['shuffle_image'], (self.size, self.size), interpolation=self.interp)

        if 'shuffle_mask_1' in data:
            data['shuffle_mask_1'] = F.resize(data['shuffle_mask_1'], (self.size, self.size), interpolation=self.interp)

        if 'shuffle_mask_2' in data:
            data['shuffle_mask_2'] = F.resize(data['shuffle_mask_2'], (self.size, self.size), interpolation=self.interp)



TrainTransform = Compose([
    ColorJitter(
        COLOR_JITTER_BRIGHTNESS,
        COLOR_JITTER_CONTRAST,
        COLOR_JITTER_SATURATION,
        COLOR_JITTER_HUE),
    RandomGrayscale(GRAYSCALE_PROBABILITY),
    ToZeroOne(),
    Normalize(NORMALIZE_RGB_MEAN, NORMALIZE_RGB_STD),
])

TestTransform = Compose([
    ToZeroOne(),
    Normalize(NORMALIZE_RGB_MEAN, NORMALIZE_RGB_STD),
])

RealWorldTestTransfrom = Compose([
    Resize(IMAGE_SIZE),
    ToZeroOne(),
    Normalize(NORMALIZE_RGB_MEAN, NORMALIZE_RGB_STD),
])

RealWorldFigTransfrom = Compose([
    Resize(IMAGE_SIZE),
    ToZeroOne(),
])