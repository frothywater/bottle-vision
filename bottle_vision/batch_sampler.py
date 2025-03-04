import logging
import math

import numpy as np
from torch.utils.data import BatchSampler

logger = logging.getLogger("bottle_vision")


class BalancedClassBatchSampler(BatchSampler):
    """Sample classes uniformly, while cycling through samples within each class.
    Iteration ends when the longest class is exhausted. Minor classes are oversampled.
    """

    def __init__(self, indices_dict: dict, classes_per_batch: int, samples_per_class: int):
        # Indices dictionary: label -> indices
        self.indices_dict = indices_dict
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class

        # Max iterations is on the longest class, drop last
        max_samples = max(len(indices) for indices in indices_dict.values())
        max_samples = math.floor(max_samples / samples_per_class) * samples_per_class
        self.max_samples = max_samples
        self.max_iter = len(indices_dict) // classes_per_batch * max_samples // samples_per_class

        logger.info(f"{max_samples=}, {self.max_iter=}, {classes_per_batch=}, {samples_per_class=}")

        # Build shuffled and cycled indices list for each class
        result = {}
        for label, indices in indices_dict.items():
            cycled_indices = np.random.permutation(indices).tolist()
            num_cycles = math.ceil(max_samples / len(indices))
            for _ in range(num_cycles - 1):
                cycled_indices += np.random.permutation(indices).tolist()
            result[label] = cycled_indices[:max_samples]
        self.indices = result

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        # Permute classes
        classes = list(self.indices.keys())

        # Proceed along sample axis
        for sample_offset in range(0, self.max_samples, self.samples_per_class):
            np.random.shuffle(classes)

            # and then along class axis
            for class_offset in range(0, len(classes), self.classes_per_batch):
                batch = []
                labels = []
                for class_idx in range(class_offset, class_offset + self.classes_per_batch):
                    label = classes[class_idx]
                    batch += self.indices[label][sample_offset : sample_offset + self.samples_per_class]
                    labels.append(label)
                logger.debug(
                    f"sample {sample_offset}-{sample_offset + self.samples_per_class}/{self.max_samples}, "
                    f"class {class_offset}-{class_offset + self.classes_per_batch}/{len(classes)}: "
                    f"{' '.join(labels)}"
                )
                yield batch
