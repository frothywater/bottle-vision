import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.functional import InterpolationMode


class PadSquare(T.Transform):
    fill: int

    def __init__(self, fill: int = 0):
        self.fill = fill
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h == w:
            return x
        size = max(h, w)
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        return T.functional.pad(x, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


class CustomResize(T.Transform):
    size: int
    interpolation: InterpolationMode

    def __init__(self, size: int, interpolation: InterpolationMode):
        self.size = size
        self.interpolation = interpolation
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make sure the longer side is resized to the target size
        h, w = x.shape[-2:]
        if h > w:
            new_h = self.size
            new_w = int(w * self.size / h)
        else:
            new_w = self.size
            new_h = int(h * self.size / w)
        return T.functional.resize(x, (new_h, new_w), interpolation=self.interpolation)


class CustomRandomCrop(T.Transform):
    min_ratio: float
    max_ratio: float

    def __init__(self, min_ratio: float, max_ratio: float):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
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
        return x.flip(0)


def get_plain_transforms(image_size: int, mean: list[float], std: list[float]):
    return T.Compose(
        [
            T.ToImage(),
            CustomResize(size=image_size, interpolation=InterpolationMode.BICUBIC),
            PadSquare(fill=255),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
            RGBToBGR(),
        ]
    )


def get_content_transforms(image_size: int, mean: list[float], std: list[float]):
    # For content, use subtler augmentations to preserve content
    return T.Compose(
        [
            T.ToImage(),
            CustomResize(size=image_size, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.25),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            PadSquare(fill=255),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
            RGBToBGR(),
        ]
    )


def get_style_transforms(image_size: int, mean: list[float], std: list[float]):
    # For style, use more aggressive augmentations to encourage robustness
    return T.Compose(
        [
            T.ToImage(),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # Problem: Rotation is too slow (~40ms/image)
            T.RandomRotation(90, interpolation=InterpolationMode.BILINEAR, fill=255),
            CustomRandomCrop(0.5, 1.0),
            CustomResize(size=image_size, interpolation=InterpolationMode.BICUBIC),
            PadSquare(fill=255),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
            RGBToBGR(),
        ]
    )
