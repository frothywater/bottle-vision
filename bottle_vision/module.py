import json
import logging
from functools import partial

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import AveragePrecision, MetricCollection, PrecisionRecallCurve

from bottle_vision.dataset import IllustDatasetItem

from .losses import ContrastiveLossConfig, ContrastiveLossParams, LossComponents, LossWeights
from .model import IllustEmbeddingModel, ModelOutput

logger = logging.getLogger("bottle_vision")


def log_dict(stage: str, losses: LossComponents, total_loss: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {
        f"{stage}/loss/quality": losses.quality,
        f"{stage}/loss/ortho": losses.ortho,
        f"{stage}/loss/total": total_loss,
    }
    for task in ["tag", "artist", "character"]:
        loss = losses.__getattribute__(task)
        if not isinstance(loss, float):
            # If a loss is not float, then it is trained by that task
            result[f"{stage}/loss/{task}"] = loss
    return result


class IllustMetricLearningModule(L.LightningModule):
    """Multi-task metric learning model for illustrations.

    Learns joint embeddings for tags, artists, and characters while maintaining
    orthogonality between different embedding spaces.
    """

    def __init__(
        self,
        # data
        num_tags: int,
        num_artists: int,
        num_characters: int,
        tag_dict_path: str,
        artist_dict_path: str,
        character_dict_path: str,
        # model
        backbone_variant: str = "vit_base_patch16_224",
        image_size: int = 448,
        tag_embed_dim: int = 512,
        artist_embed_dim: int = 128,
        character_embed_dim: int = 128,
        cls_token: bool = False,
        reg_tokens: int = 0,
        dropout: float = 0.1,
        # training
        base_lr: float = 3e-4,
        backbone_lr: float = 3e-5,
        lr_start_ratio: float = 0.1,
        lr_end_ratio: float = 0.1,
        warmup_percent: float = 0.05,
        partial_unfreeze_percent: float = 0.2,
        full_unfreeze_percent: float = 0.5,
        partial_unfreeze_layers: int = 4,
        # loss
        loss_weights: LossWeights = LossWeights(),
        tag_contrastive_config: ContrastiveLossConfig = ContrastiveLossConfig(),
        artist_contrastive_config: ContrastiveLossConfig = ContrastiveLossConfig(),
        character_contrastive_config: ContrastiveLossConfig = ContrastiveLossConfig(),
        # metric
        num_freq_bins: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model architecture
        self.model = IllustEmbeddingModel(
            num_tags=num_tags,
            num_artists=num_artists,
            num_characters=num_characters,
            backbone_variant=backbone_variant,
            image_size=image_size,
            tag_embed_dim=tag_embed_dim,
            artist_embed_dim=artist_embed_dim,
            character_embed_dim=character_embed_dim,
            cls_token=cls_token,
            reg_tokens=reg_tokens,
            dropout=dropout,
            tag_temp=tag_contrastive_config["initial_temp"],
            artist_temp=artist_contrastive_config["initial_temp"],
            character_temp=character_contrastive_config["initial_temp"],
        )

        # Load and process class frequencies
        self.tag_freqs = self._get_class_frequencies(tag_dict_path)
        self.artist_freqs = self._get_class_frequencies(artist_dict_path)
        self.character_freqs = self._get_class_frequencies(character_dict_path)

        # Get frequency bins
        self.tag_bins = self._get_frequency_bins(self.tag_freqs, num_freq_bins)
        self.artist_bins = self._get_frequency_bins(self.artist_freqs, num_freq_bins)
        self.character_bins = self._get_frequency_bins(self.character_freqs, num_freq_bins)

        # Initialize metrics
        self.val_metrics = self._create_metrics("val")
        # self.test_metrics = self._create_metrics("test")

        self.partial_unfroze = False
        self.full_unfroze = False

    def configure_model(self):
        self.model = torch.compile(self.model)
        torch.set_float32_matmul_precision("high")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _get_class_frequencies(self, dict_path: str) -> torch.Tensor:
        """Load class frequencies from indices dict."""
        with open(dict_path) as f:
            indices_dict = json.load(f)

        # Count frequencies for each class
        freqs = torch.tensor([len(indices) for indices in indices_dict.values()])
        return freqs

    def _get_frequency_bins(self, freqs: torch.Tensor, num_freq_bins: int) -> list[torch.Tensor]:
        """Split classes into several frequency bins."""
        # Calculate log frequencies
        log_freqs = torch.log(freqs + 1)  # +1 to handle zeros

        # Calculate quantiles for frequency bins
        bin_edges = torch.quantile(log_freqs, torch.linspace(0, 1, num_freq_bins + 1))
        bin_edges[-1] += 1  # Include the last element

        # Assign each class to a frequency bin
        bin_masks = []
        for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (log_freqs >= low) & (log_freqs < high)
            # Skip empty bins
            if mask.sum() > 0:
                bin_masks.append(mask)
        return bin_masks

    def training_step(self, batch: IllustDatasetItem, batch_idx: int):
        task = batch.task[0]
        params = ContrastiveLossParams.from_config(self.hparams[f"{task}_contrastive_config"])
        output: ModelOutput = self.model.forward_task(
            x=batch.image,
            task=task,
            labels=batch.__getattribute__(f"{task}_label"),
            masks=batch.__getattribute__(f"{task}_mask"),
            score=batch.score,
            contrastive_params=params,
        )

        # Compute total loss
        total_loss = output.losses.weighted_sum(self.hparams.loss_weights)
        if self.trainer.accumulate_grad_batches > 1:
            total_loss /= self.trainer.accumulate_grad_batches

        # Log all metrics
        self.log_dict(log_dict("train", output.losses, total_loss), batch_size=batch.image.shape[0])

        return total_loss

    def validation_step(self, batch: IllustDatasetItem, batch_idx: int):
        labels = dict(tag=batch.tag_label, artist=batch.artist_label, character=batch.character_label)
        masks = dict(tag=batch.tag_mask, artist=batch.artist_mask, character=batch.character_mask)
        tag_params = ContrastiveLossParams.from_config(self.hparams.tag_contrastive_config)
        artist_params = ContrastiveLossParams.from_config(self.hparams.artist_contrastive_config)
        character_params = ContrastiveLossParams.from_config(self.hparams.character_contrastive_config)
        params = dict(tag=tag_params, artist=artist_params, character=character_params)

        # Forward for all tasks
        output: ModelOutput = self.model.forward_all_tasks(
            x=batch.image,
            labels=labels,
            masks=masks,
            score=batch.score,
            contrastive_params=params,
        )

        # Compute total loss
        total_loss = output.losses.weighted_sum(self.hparams.loss_weights)

        # Log all metrics
        self.log_dict(log_dict("val", output.losses, total_loss), batch_size=batch.image.shape[0])

        # Update metrics for the task
        for task, sim_preds in output.sim_preds.items():
            mask = masks[task]
            if mask.sum() == 0:
                continue
            sim_preds = sim_preds[mask]
            # Convert float to long (due to label smoothing)
            label = labels[task][mask].round().long()

            # Take indices along class axis if doing artist task (multiclass)
            if task == "artist":
                label = label.argmax(dim=1)

            self.val_metrics[task].update(sim_preds, label)

        return total_loss

    def test_step(self, batch: IllustDatasetItem, batch_idx: int):
        labels = dict(tag=batch.tag_label, artist=batch.artist_label, character=batch.character_label)
        tag_params = ContrastiveLossParams.from_config(self.hparams.tag_contrastive_config)
        artist_params = ContrastiveLossParams.from_config(self.hparams.artist_contrastive_config)
        character_params = ContrastiveLossParams.from_config(self.hparams.character_contrastive_config)
        params = dict(tag=tag_params, artist=artist_params, character=character_params)

        # Forward for all tasks
        output: ModelOutput = self.model.forward_all_tasks(
            x=batch.image,
            labels=labels,
            score=batch.score,
            contrastive_params=params,
        )

        # Compute total loss
        total_loss = output.losses.weighted_sum(self.hparams.loss_weights)

        # Log all metrics
        self.log_dict(log_dict("test", output.losses, total_loss), batch_size=batch.image.shape[0])

        # Update metrics for the task
        for task, sim_preds in output.sim_preds.items():
            # Convert float to long (due to label smoothing)
            label = labels[task].long()
            self.test_metrics[task].update(sim_preds, label)

        return total_loss

    # ===== Optimizer and Scheduler =====

    def lr_fn(self, current_step: int, start_steps: int, warmup_steps: int, total_steps: int):
        """Linear warmup and cosine decay learning rate schedule."""
        if current_step < start_steps:
            # Not started yet
            return 0
        elif current_step < start_steps + warmup_steps:
            # Linear warmup
            progress = (current_step - start_steps) / warmup_steps
            return self.hparams.lr_start_ratio + progress * (1 - self.hparams.lr_start_ratio)
        else:
            # Cosine decay
            progress = (current_step - start_steps - warmup_steps) / (total_steps - start_steps - warmup_steps)
            return self.hparams.lr_end_ratio + (1 + np.cos(progress * np.pi)) / 2 * (1 - self.hparams.lr_end_ratio)

    def configure_optimizers(self):
        # Separate parameters for backbone and other parts of the model
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() if "backbone" not in n]

        # Create optimizers
        # 1. parameters except for backbone
        # 2. backbone parameters
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": self.hparams.base_lr},
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
            ]
        )

        # Calculate learning rate schedule
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_percent)
        backbone_start_steps = int(total_steps * self.hparams.partial_unfreeze_percent)

        # 1. parameters except for backbone: start from 0
        other_lr_fn = partial(self.lr_fn, start_steps=0, warmup_steps=warmup_steps, total_steps=total_steps)
        logger.info(f"Scheduling for other parameters: 0 -> {warmup_steps} -> {total_steps}")
        # 2. backbone parameters: start from when backbone is partially unfrozen
        backbone_lr_fn = partial(
            self.lr_fn, start_steps=backbone_start_steps, warmup_steps=warmup_steps, total_steps=total_steps
        )
        logger.info(
            f"Scheduling for backbone parameters: {backbone_start_steps} -> {backbone_start_steps + warmup_steps} -> {total_steps}"
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [other_lr_fn, backbone_lr_fn])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_fit_start(self):
        if self.trainer.ckpt_path is not None:
            logger.info(f"Loaded checkpoint from {self.trainer.ckpt_path}")
        else:
            # Load pretrained weights
            self.model.load_wd_tagger_weights(self.hparams.num_tags, self.hparams.num_characters)
        # Freeze loaded weights
        self.model.freeze_loaded_weights()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Unfreeze deeper layers progressively
        total_steps = self.trainer.estimated_stepping_batches
        current_step = self.trainer.global_step
        partial_unfreeze_steps = int(total_steps * self.hparams.partial_unfreeze_percent)
        full_unfreeze_steps = int(total_steps * self.hparams.full_unfreeze_percent)

        if not self.partial_unfroze and current_step >= partial_unfreeze_steps:
            logger.info(f"Current step: {current_step}, unfreezing deeper layers")
            self.model.unfreeze_deeper_layers(num_layers=self.hparams.partial_unfreeze_layers)
            self.partial_unfroze = True
        elif not self.full_unfroze and current_step >= full_unfreeze_steps:
            logger.info(f"Current step: {current_step}, unfreezing all layers")
            self.model.unfreeze_all()
            self.full_unfroze = True

    # ===== Metric Logging =====

    def _create_metrics(self, prefix: str, thresholds: int = 20) -> dict[str, MetricCollection]:
        """Create metrics for each task."""
        metrics = nn.ModuleDict()

        # Create metrics for each task (tag, artist, character)
        for task, num_classes in [
            ("tag", self.hparams.num_tags),
            ("artist", self.hparams.num_artists),
            ("character", self.hparams.num_characters),
        ]:
            metric_type = "multiclass" if task == "artist" else "multilabel"
            kwargs = {"num_classes": num_classes} if metric_type == "multiclass" else {"num_labels": num_classes}
            # `MetricCollection` to share underlying computation
            metrics[task] = MetricCollection(
                {
                    f"{prefix}_{task}_map": AveragePrecision(
                        task=metric_type, thresholds=thresholds, average="macro", **kwargs
                    ),
                    f"{prefix}_{task}_ap_per_class": AveragePrecision(
                        task=metric_type,
                        thresholds=thresholds,
                        average=None,  # Returns per-class scores
                        **kwargs,
                    ),
                    f"{prefix}_{task}_prc": PrecisionRecallCurve(
                        task=metric_type,
                        thresholds=thresholds,
                        **kwargs,
                    ),
                }
            )

        return metrics

    def on_validation_epoch_end(self):
        self._shared_epoch_end("val", self.val_metrics)

    def on_test_epoch_end(self):
        self._shared_epoch_end("test", self.test_metrics)

    def _shared_epoch_end(self, stage: str, metrics: dict[str, MetricCollection]):
        """Shared logic for validation and test epoch end.
        Args:
            stage: "val" or "test"
            metrics: metrics dict to compute from
        """
        # Process metrics for each task
        for task, bins in [("tag", self.tag_bins), ("artist", self.artist_bins), ("character", self.character_bins)]:
            # Get overall MAP and per-class AP scores
            metric_collection = metrics[task]
            overall_map = metric_collection[f"{stage}_{task}_map"].compute()
            ap_scores = metric_collection[f"{stage}_{task}_ap_per_class"].compute()
            precision, recall, _ = metric_collection[f"{stage}_{task}_prc"].compute()

            # Log overall MAP
            self.log(f"{stage}/metric/{task}/map/overall", overall_map)

            # Log AP distribution
            self.logger.experiment.add_histogram(f"{stage}/{task}/ap_dist", ap_scores, global_step=self.global_step)

            # Log frequency-based metrics
            for i, bin_mask in enumerate(bins):
                bin_map = ap_scores[bin_mask].mean().item()
                self.log(f"{stage}/metric/{task}/map/{i}", bin_map)

            # Create and log AP vs frequency plot
            freqs = self.__getattribute__(f"{task}_freqs")
            log_freqs = torch.log(freqs + 1)  # +1 to handle zeros
            fig = self._plot_ap_vs_frequency(ap_scores, log_freqs, f"AP vs Frequency ({task})")
            self.logger.experiment.add_figure(f"{stage}/{task}/ap_vs_freq", fig, global_step=self.global_step)
            plt.close(fig)

            # Create and log PR curves plot
            fig = self._plot_pr_curves_by_frequency(
                precision, recall, bins, f"Precision-Recall Curves by Frequency ({task})"
            )
            self.logger.experiment.add_figure(f"{stage}/{task}/pr_curves", fig, global_step=self.global_step)
            plt.close(fig)

            # Reset metrics after computing
            metric_collection.reset()

    def _plot_ap_vs_frequency(self, ap_scores: torch.Tensor, log_freqs: torch.Tensor, title: str) -> plt.Figure:
        """Create scatter plot of AP scores vs log frequencies."""
        fig, ax = plt.subplots(figsize=(10, 8))

        log_freqs = log_freqs.cpu().numpy()
        ap_scores = ap_scores.cpu().numpy()
        ax.scatter(log_freqs, ap_scores, alpha=0.5, s=10)
        ax.set_xlabel("Log Frequency")
        ax.set_ylabel("Average Precision")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        return fig

    def _plot_pr_curves_by_frequency(
        self,
        precision: torch.Tensor,  # shape: [n_classes, n_thresholds]
        recall: torch.Tensor,  # shape: [n_classes, n_thresholds]
        bins: list[torch.Tensor],
        title: str,
    ) -> plt.Figure:
        """Create PR curves plot separated by frequency bins."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Colors for each frequency bin
        color_map = plt.get_cmap("rainbow")
        colors = [color_map(i) for i in np.linspace(1, 0, len(bins))]

        # Plot mean PR curve for each frequency bin
        for i, bin_mask in enumerate(bins):
            # Macro-average on classes
            bin_precision = precision[bin_mask].mean(dim=0)
            bin_recall = recall[bin_mask].mean(dim=0)

            # Sort by recall for better visualization
            sort_idx = torch.argsort(bin_recall)
            bin_recall = bin_recall[sort_idx]
            bin_precision = bin_precision[sort_idx]

            # Calculate AUC-PR for the bin
            auc_pr = torch.trapz(bin_precision, bin_recall)

            ax.plot(
                bin_recall.cpu().numpy(),
                bin_precision.cpu().numpy(),
                color=colors[i],
                label=f"Bin {i} (AUC={auc_pr:.3f})",
                alpha=0.8,
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        return fig

    # def _log_system_stats(self):
    #     """Log system resource usage."""
    #     # GPU stats
    #     gpus = GPUtil.getGPUs()
    #     for i, gpu in enumerate(gpus):
    #         self.log(f"system/gpu{i}/memory_used_percent", gpu.memoryUtil * 100)
    #         self.log(f"system/gpu{i}/memory_used_mb", gpu.memoryUsed)
    #         self.log(f"system/gpu{i}/temperature", gpu.temperature)

    #     # CPU and RAM
    #     self.log("system/cpu_percent", psutil.cpu_percent())
    #     self.log("system/ram_percent", psutil.virtual_memory().percent)
