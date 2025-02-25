import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image


class PadSquare(T.Transform):
    fill: int

    def __init__(self, fill: int = 0):
        self.fill = fill
        super().__init__()

    def forward(self, x: Image.Image) -> Image.Image:
        w, h = x.size
        if h == w:
            return x
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        return T.functional.pad(x, (pad_w, pad_h), fill=self.fill)


class CustomRandomCrop(T.Transform):
    min_ratio: float
    max_ratio: float

    def __init__(self, min_ratio: float, max_ratio: float):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        super().__init__()

    def forward(self, x: Image.Image) -> Image.Image:
        w, h = x.size
        if h > w:
            h_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            # Ensure the aspect ratio is at least the original
            w_ratio = np.random.uniform(h_ratio, self.max_ratio)
        else:
            w_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            h_ratio = np.random.uniform(w_ratio, self.max_ratio)
        crop_h = int(h * h_ratio)
        crop_w = int(w * w_ratio)
        top = np.random.randint(0, h - crop_h)
        left = np.random.randint(0, w - crop_w)
        return T.functional.crop(x, top, left, crop_h, crop_w)


class RGBToBGR(T.Transform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[[2, 1, 0], :, :]


def get_content_transforms():
    # For content, use subtler augmentations to preserve content
    return [
        T.RandomHorizontalFlip(p=0.25),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    ]


def get_style_transforms():
    # For style, use more aggressive augmentations to encourage robustness
    return [
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.RandomRotation(90, interpolation=Image.BICUBIC, fill=255),
        CustomRandomCrop(0.5, 1.0),
    ]


def get_shared_transforms(image_size: int, mean: list[float], std: list[float]):
    return [
        PadSquare(fill=255),
        T.Resize(image_size, interpolation=Image.BICUBIC),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
        RGBToBGR(),
    ]
