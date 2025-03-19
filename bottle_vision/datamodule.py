import json
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from .batch_sampler import BalancedClassBatchSampler, InterleavedBatchSampler
from .dataset import IllustDataset, TaskIllustDataset


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
        data_tasks: list[str],
        num_workers: int = 4,
        prefetch_factor: Optional[int] = None,
        image_size: int = 448,
        valid_parquet_path: Optional[str] = None,
        valid_tar_dir: Optional[str] = None,
        test_parquet_path: Optional[str] = None,
        test_tar_dir: Optional[str] = None,
        label_smoothing_eps: float = 0,
        mean: list[float] = [0.5, 0.5, 0.5],
        std: list[float] = [0.5, 0.5, 0.5],
        task_probs: Optional[dict[str, float]] = None,
        sample_cutoff: Optional[int | dict[str, int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _collate_fn(self, batch):
        # Custom collate function to handle dataclass objects
        if len(batch) == 0:
            return batch
        if batch[0] is None:
            return None
        if hasattr(batch[0], "__dataclass_fields__"):
            return type(batch[0])(
                *[self._collate_fn([getattr(d, f) for d in batch]) for f in batch[0].__dataclass_fields__]
            )
        elif isinstance(batch[0], (tuple, list)):
            return type(batch[0])(self._collate_fn(samples) for samples in zip(*batch))
        else:
            return torch.utils.data._utils.collate.default_collate(batch)

    def train_dataloader(self):
        task_dicts = {}
        if "tag" in self.hparams.data_tasks:
            with open(self.hparams.train_tag_dict_path) as f:
                tag_dict = json.load(f)
            task_dicts["tag"] = tag_dict
        if "artist" in self.hparams.data_tasks:
            with open(self.hparams.train_artist_dict_path) as f:
                artist_dict = json.load(f)
            task_dicts["artist"] = artist_dict
        if "character" in self.hparams.data_tasks:
            with open(self.hparams.train_character_dict_path) as f:
                character_dict = json.load(f)
            task_dicts["character"] = character_dict

        dataset = TaskIllustDataset(
            parquet_path=self.hparams.train_parquet_path,
            tar_dir=self.hparams.train_tar_dir,
            num_tags=self.hparams.num_tags,
            num_artists=self.hparams.num_artists,
            num_characters=self.hparams.num_characters,
            tasks=list(task_dicts.keys()),
            image_size=self.hparams.image_size,
            mean=self.hparams.mean,
            std=self.hparams.std,
            label_smoothing_eps=self.hparams.label_smoothing_eps,
        )

        # Balanced class batch sampler for training
        samplers = {}
        sample_cutoff = self.hparams.sample_cutoff
        if isinstance(sample_cutoff, int):
            sample_cutoff = {task: sample_cutoff for task in task_dicts.keys()}
        for task, indices_dict in task_dicts.items():
            samplers[task] = BalancedClassBatchSampler(
                task=task,
                indices_dict=indices_dict,
                classes_per_batch=self.hparams.classes_per_batch,
                samples_per_class=self.hparams.samples_per_class,
                sample_cutoff=sample_cutoff[task],
            )

        # Interleaved batch sampler for different tasks
        interleaved_sampler = InterleavedBatchSampler(samplers=samplers, probs=self.hparams.task_probs)

        return DataLoader(
            dataset,
            batch_sampler=interleaved_sampler,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self._collate_fn,
            pin_memory=True,
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
            tasks=self.hparams.data_tasks,
            image_size=self.hparams.image_size,
            mean=self.hparams.mean,
            std=self.hparams.std,
            label_smoothing_eps=0,
        )
        # Use default sampler (load all images sequentially)
        return DataLoader(
            dataset,
            batch_size=self.hparams.classes_per_batch * self.hparams.samples_per_class,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def predict_dataloader(self):
        if not (self.hparams.test_parquet_path and self.hparams.test_tar_dir):
            return None

        # Use plain transforms and no label smoothing
        dataset = IllustDataset(
            parquet_path=self.hparams.test_parquet_path,
            tar_dir=self.hparams.test_tar_dir,
            num_tags=self.hparams.num_tags,
            num_artists=self.hparams.num_artists,
            num_characters=self.hparams.num_characters,
            tasks=self.hparams.data_tasks,
            image_size=self.hparams.image_size,
            mean=self.hparams.mean,
            std=self.hparams.std,
            label_smoothing_eps=0,
        )
        # Use default sampler (load all images sequentially)
        return DataLoader(
            dataset,
            batch_size=self.hparams.classes_per_batch * self.hparams.samples_per_class,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
