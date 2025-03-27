import json
import logging
from functools import partial
from typing import Literal, Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import AveragePrecision, MetricCollection, PrecisionRecallCurve
from torchmetrics.classification import (
    MultilabelCoverageError,
    MultilabelRankingAveragePrecision,
    MultilabelRankingLoss,
)

from bottle_vision.dataset import IllustDatasetItem
from bottle_vision.lora import apply_lora

from .losses import ContrastiveLossConfig, ContrastiveLossParams, LossWeights
from .model import IllustEmbeddingModel, ModelOutput

logger = logging.getLogger("bottle_vision")


def log_dict(stage: str, output: ModelOutput, total_loss: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {f"{stage}/loss/total": total_loss}
    for key, value in output.losses.__dict__.items():
        if isinstance(value, torch.Tensor):
            result[f"{stage}/loss/{key}"] = value

    for key, value in output.contrast_loss_stats.items():
        if isinstance(value, torch.Tensor):
            result[f"{stage}/loss/{key}/contrast"] = value
    for key, value in output.central_loss_stats.items():
        if isinstance(value, torch.Tensor):
            result[f"{stage}/loss/{key}/central"] = value

    return result


class IllustMetricLearningModule(L.LightningModule):
    """Multi-task metric learning model for illustrations.

    Learns joint embeddings for tags, artists, and characters.
    """

    def __init__(
        self,
        # data
        num_tags: int,
        num_artists: int,
        num_characters: int,
        tag_freq_path: str,
        artist_freq_path: str,
        character_freq_path: str,
        tasks: list[str],
        # model
        backbone_variant: str = "vit_base_patch16_224",
        image_size: int = 448,
        tag_embed_dim: int = 512,
        artist_embed_dim: int = 128,
        character_embed_dim: int = 128,
        skip_head: bool = False,
        cls_token: bool = False,
        reg_tokens: int = 0,
        dropout: float = 0.1,
        # weights
        copy_wd_weights: bool = True,
        use_pretrained_backbone: bool = False,
        use_lora: bool = True,
        lora_config: dict = {},
        strict_loading: bool = False,
        # training
        base_lr: float = 3e-4,
        lr_start_ratio: float = 0.1,
        lr_end_ratio: float = 0.1,
        warmup_percent: float = 0.05,
        decay_mode: Literal["cosine", "linear"] = "cosine",
        weight_path: Optional[str] = None,
        # loss
        loss_weights: LossWeights = {},
        use_focal_loss: bool = False,
        class_reweighting: bool = False,
        class_reweighting_method: Literal["sqrt", "log"] = "sqrt",
        temp_strategy: Literal["fixed", "task", "class"] = "fixed",
        tag_contrastive_config: ContrastiveLossConfig = {},
        artist_contrastive_config: ContrastiveLossConfig = {},
        character_contrastive_config: ContrastiveLossConfig = {},
        # metric
        num_freq_bins: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.strict_loading = strict_loading
        self.loss_weights = LossWeights.create(**loss_weights)

        # Load and process class frequencies
        self.tag_freqs = self._get_class_frequencies(tag_freq_path)
        self.character_freqs = self._get_class_frequencies(character_freq_path)
        self.artist_freqs = self._get_class_frequencies(artist_freq_path)

        # Get class weights
        if class_reweighting:
            self.tag_weights = self._get_class_weights(self.tag_freqs, class_reweighting_method)
            self.artist_weights = self._get_class_weights(self.artist_freqs, class_reweighting_method)
            self.character_weights = self._get_class_weights(self.character_freqs, class_reweighting_method)

        # Get frequency bins
        self.tag_bins = self._get_frequency_bins(self.tag_freqs, num_freq_bins)
        self.artist_bins = self._get_frequency_bins(self.artist_freqs, num_freq_bins)
        self.character_bins = self._get_frequency_bins(self.character_freqs, num_freq_bins)

        # Prepare temperature and margin values
        tag_contrastive_config = ContrastiveLossConfig.create(**tag_contrastive_config)
        artist_contrastive_config = ContrastiveLossConfig.create(**artist_contrastive_config)
        character_contrastive_config = ContrastiveLossConfig.create(**character_contrastive_config)

        tag_temp, tag_margin = self._get_temp_margin(tag_contrastive_config, self.tag_freqs)
        artist_temp, artist_margin = self._get_temp_margin(artist_contrastive_config, self.artist_freqs)
        character_temp, character_margin = self._get_temp_margin(character_contrastive_config, self.character_freqs)

        self.tag_params = ContrastiveLossParams.from_config(tag_contrastive_config, num_tags, tag_margin)
        self.artist_params = ContrastiveLossParams.from_config(artist_contrastive_config, num_artists, artist_margin)
        self.character_params = ContrastiveLossParams.from_config(
            character_contrastive_config, num_characters, character_margin
        )
        logger.info(
            f"Number of negative classes: tag={self.tag_params.num_negatives}, artist={self.artist_params.num_negatives}, character={self.character_params.num_negatives}"
        )

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
            skip_head=skip_head,
            cls_token=cls_token,
            reg_tokens=reg_tokens,
            dropout=dropout,
            tag_temp=tag_temp,
            artist_temp=artist_temp,
            character_temp=character_temp,
            tasks=tasks,
            temp_strategy=temp_strategy,
            use_pretrained_backbone=use_pretrained_backbone,
        )

        # Initialize model weights
        self.init_model()

        # Initialize metrics
        self.val_metrics = self._create_metrics("val")
        if "artist" in tasks:
            self.artist_ranking_metrics = self._create_artist_ranking_metrics("val")

    # ===== Model Configuration & Checkpointing =====

    def init_model(self):
        # 1. Copy WD weights
        if self.hparams.copy_wd_weights:
            self.model.load_wd_tagger_weights(self.hparams.num_tags, self.hparams.num_characters)

        # 2. Configure LoRA
        if self.hparams.use_lora:
            lora_params = dict(
                target_modules=["attn.qkv"],
                layers_pattern="blocks",
                # use_rslora=True,
            )
            lora_params.update(self.hparams.lora_config)

            # Manually add norm modules to train
            norm_modules = ["backbone.norm"]
            if "layers_to_transform" in lora_params:
                layer_indices = lora_params["layers_to_transform"]
                if isinstance(layer_indices, int):
                    layer_indices = [layer_indices]
                for i in layer_indices:
                    norm_modules.append(f"backbone.blocks.{i}.norm1")
                    norm_modules.append(f"backbone.blocks.{i}.norm2")

            lora_params["modules_to_save"] = self.model.trainable_module_names + norm_modules

            self.model, trainable_keys = apply_lora(self.model, **lora_params)

            logger.info(f"LoRA trainable modules: {', '.join(trainable_keys)}")

        # 3. torch.compile
        self.model = torch.compile(self.model)

        # 4. (optional) Load custom weights (not strict)
        if self.hparams.weight_path is not None:
            # Load weights
            state_dict = torch.load(self.hparams.weight_path)["state_dict"]

            result = self.model.load_state_dict(state_dict, strict=False)

            logger.info(f"Loaded weights from {self.hparams.weight_path}")
            logger.debug(f"All keys: {', '.join(state_dict.keys())}")
            if result.missing_keys:
                logger.debug(f"Missing keys: {', '.join(result.missing_keys)}")
            if result.unexpected_keys:
                logger.debug(f"Unexpected keys: {', '.join(result.unexpected_keys)}")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.use_lora:
            checkpoint["state_dict"] = self._get_peft_model_state_dict()

        logger.debug(f"Saving checkpoint, keys: {', '.join(checkpoint['state_dict'].keys())}")
        return checkpoint

    def _get_peft_model_state_dict(self):
        state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param
        return state_dict

    # ===== Training and Validation Steps =====

    def training_step(self, batch: IllustDatasetItem, batch_idx: int):
        task = batch.task[0]
        params: ContrastiveLossParams = self.__getattribute__(f"{task}_params")
        if isinstance(params.margin, torch.Tensor) and params.margin.device != batch.image.device:
            params = ContrastiveLossParams(
                margin=params.margin.to(batch.image.device),
                central_weight=params.central_weight,
            )
            self.__setattr__(f"{task}_params", params)

        if self.hparams.class_reweighting:
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
            class_weights=class_weights if self.hparams.class_reweighting else None,
            use_focal_loss=self.hparams.use_focal_loss,
        )

        # Compute total loss
        total_loss = output.losses.weighted_sum(self.loss_weights)
        if self.trainer.accumulate_grad_batches > 1:
            total_loss /= self.trainer.accumulate_grad_batches

        # Log all metrics
        self.log_dict(log_dict("train", output, total_loss), batch_size=batch.image.shape[0])
        if self.global_step % 100 == 0:
            self._log_temperatures()

        return total_loss

    def validation_step(self, batch: IllustDatasetItem, batch_idx: int):
        labels = {}
        masks = {}
        params = {}
        if batch.tag_label is not None and "tag" in self.hparams.tasks:
            labels["tag"] = batch.tag_label
            masks["tag"] = batch.tag_mask
            params["tag"] = self.tag_params
        if batch.artist_label is not None and "artist" in self.hparams.tasks:
            labels["artist"] = batch.artist_label
            masks["artist"] = batch.artist_mask
            params["artist"] = self.artist_params
        if batch.character_label is not None and "character" in self.hparams.tasks:
            labels["character"] = batch.character_label
            masks["character"] = batch.character_mask
            params["character"] = self.character_params

        for k, p in params.items():
            if isinstance(p.margin, torch.Tensor) and p.margin.device != batch.image.device:
                params[k].margin = p.margin.to(batch.image.device)

        # Forward for all tasks
        output: ModelOutput = self.model.forward_all_tasks(
            x=batch.image,
            labels=labels,
            masks=masks,
            score=batch.score,
            contrastive_params=params,
            use_focal_loss=self.hparams.use_focal_loss,
        )

        # Compute total loss
        total_loss = output.losses.weighted_sum(self.loss_weights)

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
                multi_label = label
                self.artist_ranking_metrics.update(prob_preds, multi_label)
                label = label.argmax(dim=1)

            self.val_metrics[task].update(prob_preds, label)

        return total_loss

    def predict_step(self, batch: IllustDatasetItem, batch_idx: int):
        outputs = self.model(batch.image)
        filenames = batch.filename
        return outputs, filenames

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
            progress = (current_step - start_steps - warmup_steps) / (total_steps - start_steps - warmup_steps)
            if self.hparams.decay_mode == "linear":
                # Linear decay
                return 1 - progress * (1 - self.hparams.lr_end_ratio)
            else:
                # Cosine decay
                return self.hparams.lr_end_ratio + (1 + np.cos(progress * np.pi)) / 2 * (1 - self.hparams.lr_end_ratio)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.hparams.base_lr)

        # Calculate learning rate schedule
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_percent)
        lr_fn = partial(self.lr_fn, start_steps=0, warmup_steps=warmup_steps, total_steps=total_steps)
        logger.info(f"LR Schedule: 0 -> {warmup_steps} -> {total_steps}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ===== Helpers =====

    def _get_class_frequencies(self, freq_path: str) -> torch.Tensor:
        """Load class frequencies from frequency dict."""
        with open(freq_path) as f:
            freq_dict = json.load(f)

        # Frequencies for each class (0-1 values)
        freqs = torch.tensor(list(freq_dict.values()))
        return freqs

    def _get_normalized_frequencies(self, freqs: torch.Tensor) -> torch.Tensor:
        # find the first non-zero frequency
        f_min = freqs[freqs > 0].min()
        f_max = freqs.max()
        freqs = freqs.clamp_min(f_min)
        return (freqs.log() - f_min.log()) / (f_max.log() - f_min.log())

    def _get_frequency_bins(self, freqs: torch.Tensor, num_freq_bins: int) -> list[torch.Tensor]:
        """Split classes into several frequency bins."""
        # Calculate quantiles for frequency bins
        bin_edges = torch.quantile(freqs, torch.linspace(0, 1, num_freq_bins + 1))
        bin_edges[-1] += 1  # Include the last element

        # Assign each class to a frequency bin
        bin_masks = []
        for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Mask out empty classes
            mask = (freqs >= low) & (freqs < high) & (freqs > 0)
            # Skip empty bins
            if mask.sum() > 0:
                bin_masks.append(mask)
        return bin_masks

    def _get_class_weights(
        self, freqs: torch.Tensor, method: Literal["sqrt", "log"], eps: float = 1e-4
    ) -> torch.Tensor:
        if method == "sqrt":
            result = 1 / torch.sqrt(freqs + eps)
            logger.info(
                f"Using sqrt class reweighting: min={result.min().item():.3f}, max={result.max().item():.3f}, mean={result.mean().item():.3f}"
            )
            return result
        elif method == "log":
            result = -torch.log(freqs + eps)
            logger.info(
                f"Using log class reweighting: min={result.min().item():.3f}, max={result.max().item():.3f}, mean={result.mean().item():.3f}"
            )
            return result
        else:
            raise ValueError(f"Invalid class reweighting method: {method}")

    def _get_temp_margin(
        self, config: ContrastiveLossConfig, freqs: torch.Tensor
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """If given min and max temperature/margin in the config, make temp/margin per class."""
        norm_freqs = self._get_normalized_frequencies(freqs)

        if config["min_temp"] is not None and config["max_temp"] is not None:
            temp = config["min_temp"] + norm_freqs * (config["max_temp"] - config["min_temp"])
            logger.info(
                f"Using per-class temperature: min={temp.min().item():.3f}, max={temp.max().item():.3f}, mean={temp.mean().item():.3f}"
            )
        else:
            temp = config["temp"]

        if config["min_margin"] is not None and config["max_margin"] is not None:
            margin = config["min_margin"] + norm_freqs * (config["max_margin"] - config["min_margin"])
            logger.info(
                f"Using per-class margin: min={margin.min().item():.3f}, max={margin.max().item():.3f}, mean={margin.mean().item():.3f}"
            )
        else:
            margin = config["margin"]

        return temp, margin

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
            collection = {
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
            if metric_type == "multilabel":
                collection[f"{prefix}_{task}_ranking_ap"] = MultilabelRankingAveragePrecision(**kwargs)
                collection[f"{prefix}_{task}_ranking_loss"] = MultilabelRankingLoss(**kwargs)
                collection[f"{prefix}_{task}_coverage_error"] = MultilabelCoverageError(**kwargs)

            metrics[task] = MetricCollection(collection)

        return metrics

    def _create_artist_ranking_metrics(self, prefix: str) -> MetricCollection:
        """Calling signature for multi-label ranking metrics is different from multi-class metrics."""
        task = "artist"
        num_labels = self.hparams.num_artists
        return MetricCollection(
            {
                f"{prefix}_{task}_ranking_ap": MultilabelRankingAveragePrecision(num_labels=num_labels),
                f"{prefix}_{task}_ranking_loss": MultilabelRankingLoss(num_labels=num_labels),
                f"{prefix}_{task}_coverage_error": MultilabelCoverageError(num_labels=num_labels),
            }
        )

    def _log_temperatures(self):
        if "tag" in self.hparams.tasks:
            self.log("model/tag/temperature", self.model.tag_temp.mean())
            if self.hparams.temp_strategy == "class":
                self.logger.experiment.add_histogram(
                    "model/tag/temperature_hist", self.model.tag_temp.value(), global_step=self.global_step
                )
        if "artist" in self.hparams.tasks:
            self.log("model/artist/temperature", self.model.artist_temp.mean())
            if self.hparams.temp_strategy == "class":
                self.logger.experiment.add_histogram(
                    "model/artist/temperature_hist", self.model.artist_temp.value(), global_step=self.global_step
                )
        if "character" in self.hparams.tasks:
            self.log("model/character/temperature", self.model.character_temp.mean())
            if self.hparams.temp_strategy == "class":
                self.logger.experiment.add_histogram(
                    "model/character/temperature_hist", self.model.character_temp.value(), global_step=self.global_step
                )

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

            if task != "artist":
                ranking_ap = metric_collection[f"{stage}_{task}_ranking_ap"].compute()
                ranking_loss = metric_collection[f"{stage}_{task}_ranking_loss"].compute()
                coverage_error = metric_collection[f"{stage}_{task}_coverage_error"].compute()
            else:
                ranking_ap = self.artist_ranking_metrics[f"{stage}_{task}_ranking_ap"].compute()
                ranking_loss = self.artist_ranking_metrics[f"{stage}_{task}_ranking_loss"].compute()
                coverage_error = self.artist_ranking_metrics[f"{stage}_{task}_coverage_error"].compute()

            # Log overall scalar metrics
            self.log(f"{stage}/metric/{task}/map/overall", overall_map)
            self.log(f"{stage}/metric/{task}/ranking/ap", ranking_ap)
            self.log(f"{stage}/metric/{task}/ranking/loss", ranking_loss)
            self.log(f"{stage}/metric/{task}/ranking/coverage_error", coverage_error)

            # Log AP distribution
            self.logger.experiment.add_histogram(f"{stage}/{task}/ap_dist", ap_scores, global_step=self.global_step)

            # Log frequency-based metrics
            for i, bin_mask in enumerate(bins):
                bin_map = ap_scores[bin_mask].mean().item()
                self.log(f"{stage}/metric/{task}/map/{i}", bin_map)

            # Create and log AP vs frequency plot
            freqs = self.__getattribute__(f"{task}_freqs")
            fig = self._plot_ap_vs_frequency(ap_scores, freqs, f"AP vs Frequency ({task})")
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

    def _plot_ap_vs_frequency(self, ap_scores: torch.Tensor, freqs: torch.Tensor, title: str) -> plt.Figure:
        """Create scatter plot of AP scores vs log frequencies."""
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = freqs > 0
        freqs = self._get_normalized_frequencies(freqs[mask])
        ap_scores = ap_scores[mask]
        freqs = freqs.cpu().numpy()
        ap_scores = ap_scores.cpu().numpy()

        ax.scatter(freqs, ap_scores, alpha=0.5, s=10)
        ax.set_xlabel("Normalized Frequency")
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
