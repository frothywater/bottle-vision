import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pyarrow.parquet as pq
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class IllustDatasetItem:
    image: torch.Tensor
    tag_label: torch.Tensor
    artist_label: torch.Tensor
    character_label: torch.Tensor
    score: torch.Tensor


class IllustDataset(Dataset):
    """Dataset for illustration data with tags, artists, characters, and quality scores.

    Implements memory-efficient loading of Parquet files by using row groups.

    Args:
        parquet_path: Path to the Parquet file containing metadata
        tar_dir: Directory containing image tar files
        num_tags: Number of possible tags
        num_artists: Number of possible artists
        num_characters: Number of possible characters
        transform: Optional transforms to apply to images
    """

    def __init__(
        self,
        parquet_path: str,
        tar_dir: str,
        num_tags: int,
        num_artists: int,
        num_characters: int,
        label_smoothing_eps: float,
        transform: Optional[T.Transform] = None,
    ):
        super().__init__()
        self.transform = transform
        self.table = pq.read_table(parquet_path)
        self.tar_dir = tar_dir
        self.num_tags = num_tags
        self.num_artists = num_artists
        self.num_characters = num_characters
        self.label_smoothing_eps = label_smoothing_eps

    def process_image(self, img: Image.Image) -> Image.Image:
        # Convert to RGB or RGBA
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")
        # Convert RGBA to RGB with white background
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, 255)
            img = Image.alpha_composite(bg, img).convert("RGB")
        return img

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> IllustDatasetItem:
        # Get row
        row = self.table.slice(idx, 1)

        # Extract data
        tar_filename = row["tar"][0].as_py()
        offset = row["offset"][0].as_py()
        filesize = row["size"][0].as_py()
        tar_path = os.path.join(self.tar_dir, tar_filename)
        with open(tar_path, "rb") as f:
            f.seek(offset)
            data = f.read(filesize)
            image = Image.open(BytesIO(data))

        image = self.process_image(image)
        if self.transform is not None:
            image = self.transform(image)

        # Tag multi-hot vector with label smoothing
        eps = self.label_smoothing_eps
        tags = row["tags"][0].as_py()
        tag_label = torch.full((self.num_tags,), eps / (self.num_tags - len(tags)), dtype=torch.float32)
        for tag in tags:
            tag_label[tag] = 1.0 - eps / len(tags)

        # Artist one-hot vector with label smoothing
        artist = row["artist"][0].as_py()
        if artist is not None:
            artist_label = torch.full((self.num_artists,), eps / (self.num_artists - 1), dtype=torch.float32)
            artist_label[artist] = 1.0 - eps
        else:
            artist_label = torch.zeros((self.num_artists,), dtype=torch.float32)

        # Character multi-hot vector with label smoothing
        characters = row["characters"][0].as_py()
        character_label = torch.full(
            (self.num_characters,), eps / (self.num_characters - len(characters)), dtype=torch.float32
        )
        for character in characters:
            character_label[character] = 1.0 - eps / len(characters)

        # Quality score
        score = row["score"][0].as_py()
        score = torch.tensor(score, dtype=torch.float32)

        return IllustDatasetItem(
            image=image,
            tag_label=tag_label,
            artist_label=artist_label,
            character_label=character_label,
            score=score,
        )
