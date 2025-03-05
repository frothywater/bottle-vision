import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader

from .dataset import IllustDatasetItem


@dataclass
class InterleavedDataItem:
    task: str
    data: IllustDatasetItem


class InterleavedDataLoader(Iterator):
    """
    Interleave multiple dataloaders.
    Stop when the longest loader is exhausted and repeat the rest loaders.
    If given proabilities for each loader, loaders will be sampled according to the probabilities.
    """

    def __init__(self, loaders: dict[str, DataLoader], probs: dict[str, float] = None):
        self.loaders = loaders
        self.max_len = max(len(loader) for loader in loaders.values())
        if probs is not None:
            probs = {key: prob / sum(probs.values()) for key, prob in probs.items()}
            self.total_batches = max(int(len(loader) / probs[key]) for key, loader in loaders.items())
        else:
            self.total_batches = self.max_len * len(self.loaders)
        self.probs = probs

        self.iterators = {}
        for key in self.loaders:
            loader = self.loaders[key]
            if len(loader) < self.max_len:
                self.iterators[key] = itertools.cycle(iter(loader))
            else:
                self.iterators[key] = iter(loader)

        self.current_batch = 0
        self.keys = list(self.loaders.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.total_batches:
            raise StopIteration

        if self.probs is not None:
            key = np.random.choice(self.keys, p=[self.probs[key] for key in self.keys])
        else:
            key = self.keys[self.current_batch % len(self.keys)]
        self.current_batch += 1

        return InterleavedDataItem(task=key, data=next(self.iterators[key]))

    def __len__(self):
        return self.total_batches
