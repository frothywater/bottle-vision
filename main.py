import logging

import torch
from lightning.pytorch.cli import LightningCLI

from bottle_vision.datamodule import IllustDataModule
from bottle_vision.module import IllustMetricLearningModule

if __name__ == "__main__":
    logger = logging.getLogger("bottle_vision")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)

    torch.set_float32_matmul_precision("high")

    cli = LightningCLI(IllustMetricLearningModule, IllustDataModule)
