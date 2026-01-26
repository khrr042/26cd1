# encoding: utf-8
import torchvision.transforms as T
from PIL import Image
import random
import torch


class RandomErasing(object):
    def __init__(self, probability=0.5, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        if not isinstance(img, torch.Tensor):
            return img

        c, h, w = img.size()
        area = h * w

        for _ in range(100):
            target_area = random.uniform(0.02, 0.4) * area
            aspect_ratio = random.uniform(0.3, 3.3)

            erase_h = int(round((target_area * aspect_ratio) ** 0.5))
            erase_w = int(round((target_area / aspect_ratio) ** 0.5))

            if erase_w < w and erase_h < h:
                x1 = random.randint(0, h - erase_h)
                y1 = random.randint(0, w - erase_w)
                img[:, x1:x1 + erase_h, y1:y1 + erase_w] = torch.tensor(
                    self.mean
                ).view(c, 1, 1)
                return img

        return img


def build_transforms(cfg, mode=None, is_train=None):
    if mode is not None:
        train = mode == 'train'
    elif is_train is not None:
        train = is_train
    else:
        raise ValueError('Either mode or is_train must be specified')

    if hasattr(cfg.INPUT, 'IMG_HEIGHT'):
        h = cfg.INPUT.IMG_HEIGHT
        w = cfg.INPUT.IMG_WIDTH
    else:
        if train:
            h, w = cfg.INPUT.SIZE_TRAIN
        else:
            h, w = cfg.INPUT.SIZE_TEST

    normalize = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD
    )

    if train:
        transform = T.Compose([
            T.Resize((h, w), interpolation=Image.BILINEAR),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop((h, w)),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.15,
                saturation=0.0,
                hue=0.0
            ),
            T.ToTensor(),
            normalize,
            RandomErasing(
                probability=cfg.INPUT.RE_PROB,
                mean=cfg.INPUT.PIXEL_MEAN
            )
        ])

    else:
        transform = T.Compose([
            T.Resize((h, w), interpolation=Image.BILINEAR),
            T.ToTensor(),
            normalize
        ])

    return transform
