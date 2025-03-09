import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transform import get_content_transforms, get_plain_transforms, get_style_transforms

logger = logging.getLogger("bottle_vision")


@dataclass
class IllustDatasetItem:
    filename: str
    image: torch.Tensor
    score: torch.Tensor
    task: Optional[str] = None
    tag_label: Optional[torch.Tensor] = None
    artist_label: Optional[torch.Tensor] = None
    character_label: Optional[torch.Tensor] = None
    tag_mask: Optional[torch.Tensor] = None
    artist_mask: Optional[torch.Tensor] = None
    character_mask: Optional[torch.Tensor] = None


class IllustDataset(Dataset):
    """Dataset for illustration data with tags, artists, characters, and quality scores.

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
        image_size: int = 448,
        mean: list[float] = [0.5, 0.5, 0.5],
        std: list[float] = [0.5, 0.5, 0.5],
        label_smoothing_eps: float = 0,
    ):
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tar_dir = tar_dir
        self.num_tags = num_tags
        self.num_artists = num_artists
        self.num_characters = num_characters
        self.label_smoothing_eps = label_smoothing_eps
        self.plain_transforms = get_plain_transforms(image_size, mean, std)

        # Initialize a cache for opened tar files.
        self.tar_file_cache = {}

    def _fetch_image_data(self, row) -> Image.Image:
        # Extract data
        tar_filename = row["tar"][0].as_py()
        offset = row["offset"][0].as_py()
        filesize = row["size"][0].as_py()
        tar_path = os.path.join(self.tar_dir, tar_filename)

        # Check cache; if not opened, open and store the file handle.
        if tar_filename not in self.tar_file_cache:
            self.tar_file_cache[tar_filename] = open(tar_path, "rb")

        f = self.tar_file_cache[tar_filename]
        f.seek(offset)
        data = f.read(filesize)
        image = Image.open(BytesIO(data))
        image.load()
        return image

    def _process_image(self, img: Image.Image) -> Image.Image:
        # Convert to RGB or RGBA
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")
        # Convert RGBA to RGB with white background
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, 255)
            img = Image.alpha_composite(bg, img).convert("RGB")
        return img

    def _transform_image(self, image: Image.Image, task: str = None) -> torch.Tensor:
        return self.plain_transforms(image)

    def _tag_label(self, row) -> torch.Tensor:
        # Tag multi-hot vector with label smoothing
        eps = self.label_smoothing_eps
        tags = row["tags"][0].as_py()
        tag_label = torch.full((self.num_tags,), eps / (self.num_tags - len(tags)), dtype=torch.float32)
        for tag in tags:
            tag_label[tag] = 1.0 - eps / len(tags)
        if len(tags) > 0:
            tag_mask = torch.tensor(True, dtype=torch.bool)
        else:
            tag_mask = torch.tensor(False, dtype=torch.bool)
        return tag_label, tag_mask

    def _artist_label(self, row) -> torch.Tensor:
        # Artist one-hot vector with label smoothing
        eps = self.label_smoothing_eps
        artist = row["artist"][0].as_py()
        artist_label = torch.full((self.num_artists,), eps / (self.num_artists - 1), dtype=torch.float32)
        if artist is not None:
            artist_label[artist] = 1.0 - eps
            artist_mask = torch.tensor(True, dtype=torch.bool)
        else:
            artist_label = torch.zeros((self.num_artists,), dtype=torch.float32)
            artist_mask = torch.tensor(False, dtype=torch.bool)
        return artist_label, artist_mask

    def _character_label(self, row) -> torch.Tensor:
        # Character multi-hot vector with label smoothing
        eps = self.label_smoothing_eps
        characters = row["characters"][0].as_py()
        character_label = torch.full(
            (self.num_characters,), eps / (self.num_characters - len(characters)), dtype=torch.float32
        )
        for character in characters:
            character_label[character] = 1.0 - eps / len(characters)
        if len(characters) > 0:
            character_mask = torch.tensor(True, dtype=torch.bool)
        else:
            character_mask = torch.tensor(False, dtype=torch.bool)
        return character_label, character_mask

    def _quality_score(self, row) -> torch.Tensor:
        # Quality score
        score = row["score"][0].as_py()
        score = torch.tensor(score, dtype=torch.float32)
        return score

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int) -> IllustDatasetItem:
        # Get row
        row = self.table.slice(idx, 1)

        image = self._fetch_image_data(row)
        image = self._process_image(image)
        image = self._transform_image(image)

        tag_label, tag_mask = self._tag_label(row)
        character_label, character_mask = self._character_label(row)
        artist_label, artist_mask = self._artist_label(row)

        return IllustDatasetItem(
            image=image,
            tag_label=tag_label,
            artist_label=artist_label,
            character_label=character_label,
            tag_mask=tag_mask,
            artist_mask=artist_mask,
            character_mask=character_mask,
            score=self._quality_score(row),
            filename=row["filename"][0].as_py(),
        )

    def __del__(self):
        # Ensure all file handles are closed when the dataset is destroyed.
        for f in self.tar_file_cache.values():
            try:
                f.close()
            except Exception:
                pass


class TaskIllustDataset(IllustDataset):
    """Dataset for illustration data with tags, artists, characters, and quality scores.

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
        tasks: list[str],
        image_size: int = 448,
        mean: list[float] = [0.5, 0.5, 0.5],
        std: list[float] = [0.5, 0.5, 0.5],
        label_smoothing_eps: float = 0,
    ):
        super().__init__(
            parquet_path=parquet_path,
            tar_dir=tar_dir,
            num_tags=num_tags,
            num_artists=num_artists,
            num_characters=num_characters,
            image_size=image_size,
            mean=mean,
            std=std,
            label_smoothing_eps=label_smoothing_eps,
        )

        self.tasks = tasks
        self.content_transforms = get_content_transforms(image_size, mean, std)
        self.style_transforms = get_style_transforms(image_size, mean, std)

    def _transform_image(self, image: Image.Image, task: str = None) -> torch.Tensor:
        if task == "artist":
            logging.debug("Applying style transforms")
            return self.style_transforms(image)
        else:
            logging.debug("Applying content transforms")
            return self.content_transforms(image)

    def __getitem__(self, idx: int) -> IllustDatasetItem:
        # Decode task from index
        task = self.tasks[idx % len(self.tasks)]
        idx //= len(self.tasks)
        logger.debug(f"Dataset task: {task}")

        # Get row
        row = self.table.slice(idx, 1)

        image = self._fetch_image_data(row)
        image = self._process_image(image)
        image = self._transform_image(image, task=task)

        item = IllustDatasetItem(
            task=task,
            image=image,
            score=self._quality_score(row),
            filename=row["filename"][0].as_py(),
        )

        if task == "tag":
            item.tag_label, item.tag_mask = self._tag_label(row)
        elif task == "artist":
            item.artist_label, item.artist_mask = self._artist_label(row)
        elif task == "character":
            item.character_label, item.character_mask = self._character_label(row)

        return item
