import json
from typing import Optional

import lightning as L
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from .batch_sampler import BalancedClassBatchSampler
from .dataloader import InterleavedDataLoader
from .dataset import IllustDataset
from .transform import get_content_transforms, get_shared_transforms, get_style_transforms


class IllustDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_parquet_path: str,
        train_tar_dir: str,
        train_tag_dict_path: str,
        train_artist_dict_path: str,
        train_character_dict_path: str,
        classes_per_batch: int,
        samples_per_class: int,
        num_tags: int,
        num_artists: int,
        num_characters: int,
        num_workers: int = 4,
        image_size: int = 448,
        valid_parquet_path: Optional[str] = None,
        valid_tar_dir: Optional[str] = None,
        label_smoothing_eps: float = 0,
        mean: list[float] = [0.5, 0.5, 0.5],
        std: list[float] = [0.5, 0.5, 0.5],
        task_probs: Optional[dict[str, float]] = None,
        sample_cutoff: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.plain_transforms = T.Compose(get_shared_transforms(image_size, mean, std))
        self.content_transforms = T.Compose(get_content_transforms() + get_shared_transforms(image_size, mean, std))
        self.style_transforms = T.Compose(get_style_transforms() + get_shared_transforms(image_size, mean, std))

    def _collate_fn(self, batch):
        # Custom collate function to handle dataclass objects
        if len(batch) == 0:
            return batch
        if hasattr(batch[0], "__dataclass_fields__"):
            return type(batch[0])(
                *[self._collate_fn([getattr(d, f) for d in batch]) for f in batch[0].__dataclass_fields__]
            )
        elif isinstance(batch[0], (tuple, list)):
            return type(batch[0])(self._collate_fn(samples) for samples in zip(*batch))
        else:
            return torch.utils.data._utils.collate.default_collate(batch)

    def train_dataloader(self):
        with open(self.hparams.train_tag_dict_path) as f:
            tag_dict = json.load(f)
        with open(self.hparams.train_artist_dict_path) as f:
            artist_dict = json.load(f)
        with open(self.hparams.train_character_dict_path) as f:
            character_dict = json.load(f)

        # Separate dataset for content (tags, characters) and style (artists)
        # Apply different data augmentation
        content_dataset = IllustDataset(
            parquet_path=self.hparams.train_parquet_path,
            tar_dir=self.hparams.train_tar_dir,
            num_tags=self.hparams.num_tags,
            num_artists=self.hparams.num_artists,
            num_characters=self.hparams.num_characters,
            transform=self.content_transforms,
            label_smoothing_eps=self.hparams.label_smoothing_eps,
        )
        style_dataset = IllustDataset(
            parquet_path=self.hparams.train_parquet_path,
            tar_dir=self.hparams.train_tar_dir,
            num_tags=self.hparams.num_tags,
            num_artists=self.hparams.num_artists,
            num_characters=self.hparams.num_characters,
            transform=self.style_transforms,
            label_smoothing_eps=self.hparams.label_smoothing_eps,
        )

        # Balanced class batch sampler for training
        tag_sampler = BalancedClassBatchSampler(
            indices_dict=tag_dict,
            classes_per_batch=self.hparams.classes_per_batch,
            samples_per_class=self.hparams.samples_per_class,
            sample_cutoff=self.hparams.sample_cutoff,
        )
        artist_sampler = BalancedClassBatchSampler(
            indices_dict=artist_dict,
            classes_per_batch=self.hparams.classes_per_batch,
            samples_per_class=self.hparams.samples_per_class,
            sample_cutoff=self.hparams.sample_cutoff,
        )
        character_sampler = BalancedClassBatchSampler(
            indices_dict=character_dict,
            classes_per_batch=self.hparams.classes_per_batch,
            samples_per_class=self.hparams.samples_per_class,
            sample_cutoff=self.hparams.sample_cutoff,
        )

        # DataLoader
        tag_loader = DataLoader(
            content_dataset,
            batch_sampler=tag_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
        )
        artist_loader = DataLoader(
            style_dataset,
            batch_sampler=artist_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
        )
        character_loader = DataLoader(
            content_dataset,
            batch_sampler=character_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
        )

        # Return interleaved dataloader for 3 tasks
        return InterleavedDataLoader(
            {"tag": tag_loader, "character": character_loader, "artist": artist_loader}, probs=self.hparams.task_probs
        )

    def val_dataloader(self):
        if not (self.hparams.valid_parquet_path and self.hparams.valid_tar_dir):
            return None

        # Use plain transforms and no label smoothing
        dataset = IllustDataset(
            parquet_path=self.hparams.valid_parquet_path,
            tar_dir=self.hparams.valid_tar_dir,
            num_tags=self.hparams.num_tags,
            num_artists=self.hparams.num_artists,
            num_characters=self.hparams.num_characters,
            transform=self.plain_transforms,
            label_smoothing_eps=0,
        )
        # Use default sampler (load all images sequentially)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.classes_per_batch * self.hparams.samples_per_class,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
        )
        return dataloader
