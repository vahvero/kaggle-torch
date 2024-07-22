"""Torchvision style transforms
"""
from torch import nn
from torchvision.transforms.functional import vflip, hflip


def vertical_flip_target(target):
    target["boxes"] = target["boxes"][:, (2, 1, 0, 3)]
    target["masks"] = vflip(target["masks"])
    return target


def horizontal_flip_target(target):
    target["boxes"] = target["boxes"][:, (0, 3, 2, 1)]
    target["masks"] = hflip(target["masks"])
    return target


class Flip(nn.Module):
    def __init__(self, propability: float = 0.5):
        super().__init__()
        self.propability = propability


class VerticalFlip(Flip):
    def forward(self, image, targets):
        if self.propability:
            image = vflip(image)
            target = vertical_flip_target(target)
        return image, targets


class HorizontalFlip(Flip):
    def forward(self, image, target):
        if self.propability:
            image = hflip(image)
            target = horizontal_flip_target(target)
        return image, target
