from collections.abc import Iterator
from dataclasses import dataclass
import math

import numpy as np
from torch.utils.data import DataLoader

from .dataset import IllustDatasetItem


@dataclass
class InterleavedDataItem:
    task: str
    data: IllustDatasetItem


def cycling_iterator(loader):
    while True:
        for batch in loader:
            yield batch


class InterleavedDataLoader(Iterator):
    """
    Interleave multiple dataloaders.
    Stop when the longest loader is exhausted and repeat the rest loaders.
    If given proabilities for each loader, loaders will be sampled according to the probabilities.
    """

    def __init__(self, loaders: dict[str, DataLoader], probs: dict[str, float] = None):
        self.loaders = loaders

        if probs is not None:
            self.probs = {key: prob / sum(probs.values()) for key, prob in probs.items()}
            self.total_batches = max(math.ceil(len(loader) / probs[key]) for key, loader in loaders.items())

            # Prepare shuffled key index list
            key_indices = []
            for i, key in enumerate(loaders.keys()):
                key_indices += [i] * math.ceil(self.total_batches * self.probs[key])
            assert len(key_indices) >= self.total_batches
            np.random.shuffle(key_indices)
            self.key_indices = key_indices
        else:
            max_len = max(len(loader) for loader in loaders.values())
            self.total_batches = max_len * len(self.loaders)
            self.probs = None

        self.iterators = {key: cycling_iterator(loader) for key, loader in self.loaders.items()}

        self.current_batch = 0
        self.keys = list(self.loaders.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.total_batches:
            raise StopIteration

        if self.probs is not None:
            key = self.keys[self.key_indices[self.current_batch]]
        else:
            key = self.keys[self.current_batch % len(self.keys)]
        self.current_batch += 1

        return InterleavedDataItem(task=key, data=next(self.iterators[key]))

    def __len__(self):
        return self.total_batches
