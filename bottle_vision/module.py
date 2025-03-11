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

from .losses import ContrastiveLossConfig, ContrastiveLossParams, LossWeights
from .model import IllustEmbeddingModel, ModelOutput

logger = logging.getLogger("bottle_vision")


def log_dict(stage: str, output: ModelOutput, total_loss: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {f"{stage}/loss/total": total_loss}
    for key, value in output.losses.__dict__.items():
        if isinstance(value, torch.Tensor):
            result[f"{stage}/loss/{key}"] = value

    for key, value in output.contrast_loss_stats.items():
        result[f"{stage}/loss/{key}/contrast"] = value
    for key, value in output.central_loss_stats.items():
        result[f"{stage}/loss/{key}/central"] = value

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
        tasks: list[str],
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
        unfreeze_schedule: dict[int, float] = {4: 0.33, 12: 0.67},
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
            tasks=tasks,
        )

        # Load and process class frequencies
        self.tag_freqs = self._get_class_frequencies(tag_dict_path)
        self.artist_freqs = self._get_class_frequencies(artist_dict_path)
        self.character_freqs = self._get_class_frequencies(character_dict_path)

        # Get class weights
        self.tag_weights = self._get_class_weights(self.tag_freqs)
        self.artist_weights = self._get_class_weights(self.artist_freqs)
        self.character_weights = self._get_class_weights(self.character_freqs)

        # Get frequency bins
        self.tag_bins = self._get_frequency_bins(self.tag_freqs, num_freq_bins)
        self.artist_bins = self._get_frequency_bins(self.artist_freqs, num_freq_bins)
        self.character_bins = self._get_frequency_bins(self.character_freqs, num_freq_bins)

        # Initialize metrics
        self.val_metrics = self._create_metrics("val")

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

    def _get_class_weights(self, freqs: torch.Tensor, alpha: float = 0.5, eps: float = 1e-4) -> torch.Tensor:
        freqs = freqs / freqs.sum()
        return (freqs + eps) ** -alpha

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
        class_weights = self.__getattribute__(f"{task}_weights")
        if class_weights.device != batch.image.device:
            class_weights = class_weights.to(batch.image.device)
            self.__setattr__(f"{task}_weights", class_weights)

        output: ModelOutput = self.model.forward_task(
            x=batch.image,
            task=task,
            labels=batch.__getattribute__(f"{task}_label"),
            masks=batch.__getattribute__(f"{task}_mask"),
            score=batch.score,
            contrastive_params=params,
            class_weights=class_weights,
        )

        # Compute total loss
        total_loss = output.losses.weighted_sum(self.hparams.loss_weights)
        if self.trainer.accumulate_grad_batches > 1:
            total_loss /= self.trainer.accumulate_grad_batches

        # Log all metrics
        self.log_dict(log_dict("train", output, total_loss), batch_size=batch.image.shape[0])
        self._log_temperatures()

        return total_loss

    def validation_step(self, batch: IllustDatasetItem, batch_idx: int):
        labels = {}
        masks = {}
        params = {}
        class_weights = {}
        if batch.tag_label is not None and "tag" in self.hparams.tasks:
            labels["tag"] = batch.tag_label
            masks["tag"] = batch.tag_mask
            params["tag"] = ContrastiveLossParams.from_config(self.hparams.tag_contrastive_config)
            class_weights["tag"] = self.tag_weights
        if batch.artist_label is not None and "artist" in self.hparams.tasks:
            labels["artist"] = batch.artist_label
            masks["artist"] = batch.artist_mask
            params["artist"] = ContrastiveLossParams.from_config(self.hparams.artist_contrastive_config)
            class_weights["artist"] = self.artist_weights
        if batch.character_label is not None and "character" in self.hparams.tasks:
            labels["character"] = batch.character_label
            masks["character"] = batch.character_mask
            params["character"] = ContrastiveLossParams.from_config(self.hparams.character_contrastive_config)
            class_weights["character"] = self.character_weights

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
        self.log_dict(log_dict("val", output, total_loss), batch_size=batch.image.shape[0])

        # Update metrics for the task
        for task, prob_preds in output.prob_preds.items():
            mask = masks[task]
            if mask.sum() == 0:
                continue
            prob_preds = prob_preds[mask]
            # Convert float to long (due to label smoothing)
            label = labels[task][mask].round().long()

            # Take indices along class axis if doing artist task (multiclass)
            if task == "artist":
                label = label.argmax(dim=1)

            self.val_metrics[task].update(prob_preds, label)

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

        # Gradual unfreezing
        total_steps = self.trainer.estimated_stepping_batches
        unfreeze_schedule = [
            (int(percent * total_steps), int(layers)) for layers, percent in self.hparams.unfreeze_schedule.items()
        ]
        self.step_to_unfrozen_layers = dict(sorted(unfreeze_schedule, key=lambda x: x[0], reverse=True))
        self.current_unfrozen_layers = 0
        logger.info(f"Unfreeze schedule: {self.step_to_unfrozen_layers}")

        # Calculate learning rate schedule
        warmup_steps = int(total_steps * self.hparams.warmup_percent)
        backbone_start_steps = min(self.step_to_unfrozen_layers.keys())

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

    def on_train_batch_start(self, batch, batch_idx):
        # Gradual unfreezing
        for step, layers in self.step_to_unfrozen_layers.items():
            if self.global_step >= step and layers > self.current_unfrozen_layers:
                self.model.unfreeze_layers(layers)
                self.current_unfrozen_layers = layers

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
            if task not in self.hparams.tasks:
                continue

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

    def _log_temperatures(self):
        if "tag" in self.hparams.tasks:
            tag_temp = self.model.tag_temp
            self.log("model/tag/temperature", tag_temp)
            # self.logger.experiment.add_histogram("model/tag/temperature_hist", tag_temp, global_step=self.global_step)
        if "artist" in self.hparams.tasks:
            artist_temp = self.model.artist_temp
            self.log("model/artist/temperature", artist_temp)
            # self.logger.experiment.add_histogram("model/artist/temperature_hist", artist_temp, global_step=self.global_step)
        if "character" in self.hparams.tasks:
            character_temp = self.model.character_temp
            self.log("model/character/temperature", character_temp)
            # self.logger.experiment.add_histogram("model/character/temperature_hist", character_temp, global_step=self.global_step)

    def on_validation_epoch_end(self):
        stage = "val"

        # Process metrics for each task
        for task, bins in [("tag", self.tag_bins), ("artist", self.artist_bins), ("character", self.character_bins)]:
            if task not in self.hparams.tasks:
                continue

            # Get overall MAP and per-class AP scores
            metric_collection = self.val_metrics[task]
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
