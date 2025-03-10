import logging
import math
from typing import Optional

import numpy as np
from torch.utils.data import BatchSampler

logger = logging.getLogger("bottle_vision")


class BalancedClassBatchSampler(BatchSampler):
    """Sample classes uniformly, while cycling through samples within each class.
    Iteration ends when the longest class is exhausted. Minor classes are oversampled.
    """

    def __init__(
        self,
        task: str,
        indices_dict: dict,
        classes_per_batch: int,
        samples_per_class: int,
        sample_cutoff: Optional[int] = None,
    ):
        # Indices dictionary: label -> indices
        self.indices_dict = indices_dict
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class

        max_samples = max(len(indices) for indices in indices_dict.values())
        if sample_cutoff:
            self.num_samples = min(sample_cutoff, max_samples)
        else:
            self.num_samples = (max_samples // samples_per_class) * samples_per_class

        self.max_iter = len(indices_dict) // classes_per_batch * self.num_samples // samples_per_class

        self.task = task
        logger.info(
            f"{task} class sampler: {len(indices_dict)=}, {self.num_samples=}, {self.max_iter=}, {classes_per_batch=}, {samples_per_class=}"
        )

    def _infinite_generator(self, indices):
        """Yield indices on the fly by cycling through a new permutation each time."""
        while True:
            for idx in np.random.permutation(indices):
                yield idx

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        # Build an on-the-fly generator for each class.
        classes = list(self.indices_dict.keys())
        generators = {label: self._infinite_generator(indices) for label, indices in self.indices_dict.items()}

        for sample_offset in range(0, self.num_samples, self.samples_per_class):
            np.random.shuffle(classes)
            for class_offset in range(0, len(classes), self.classes_per_batch):
                batch = []
                batch_labels = []
                for label in classes[class_offset : class_offset + self.classes_per_batch]:
                    batch.extend(next(generators[label]) for _ in range(self.samples_per_class))
                    batch_labels.append(label)
                logger.debug(
                    f"{self.task} class sampler: "
                    f"sample {sample_offset}-{sample_offset + self.samples_per_class}/{self.num_samples}, "
                    f"class {class_offset}-{class_offset + self.classes_per_batch}/{len(classes)}: "
                    f"{' '.join(batch_labels)}"
                )
                yield batch


class InterleavedBatchSampler(BatchSampler):
    """
    Interleave multiple samplers.
    Stop when the longest sampler is exhausted and repeat the rest samplers.
    If given proabilities for each sampler, samplers will be sampled according to the probabilities.
    """

    def __init__(self, samplers: dict[str, BatchSampler], probs: dict[str, float] = None):
        self.samplers = samplers

        if probs is not None:
            probs = {key: prob for key, prob in probs.items() if key in samplers}
            self.probs = {key: prob / sum(probs.values()) for key, prob in probs.items()}
            self.total_batches = max(math.ceil(len(sampler) / self.probs[key]) for key, sampler in samplers.items())

            # Prepare shuffled key index list
            task_indices = []
            for i, key in enumerate(samplers.keys()):
                task_indices += [i] * math.ceil(self.total_batches * self.probs[key])
            assert len(task_indices) >= self.total_batches
            np.random.shuffle(task_indices)
            self.task_indices = task_indices
        else:
            max_len = max(len(sampler) for sampler in samplers.values())
            self.total_batches = max_len * len(self.samplers)
            self.probs = None

    def _infinite_generator(self, sampler):
        while True:
            yield from sampler

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        tasks = list(self.samplers.keys())
        generators = {key: self._infinite_generator(sampler) for key, sampler in self.samplers.items()}

        for batch_idx in range(self.total_batches):
            if self.probs is not None:
                task_index = self.task_indices[batch_idx]
            else:
                task_index = batch_idx % len(tasks)
            task = tasks[task_index]

            raw_indices = next(generators[task])

            # Encode task flag
            encoded_indices = [raw_index * len(tasks) + task_index for raw_index in raw_indices]
            logging.debug(f"Interleaved sampler: task {task}")

            yield encoded_indices
