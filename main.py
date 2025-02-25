from lightning.pytorch.cli import LightningCLI

from bottle_vision.datamodule import IllustDataModule
from bottle_vision.module import IllustMetricLearningModule

if __name__ == "__main__":
    cli = LightningCLI(IllustMetricLearningModule, IllustDataModule)
