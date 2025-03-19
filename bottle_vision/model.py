import logging
from dataclasses import dataclass
from typing import Literal, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import ContrastiveLossParams, LossComponents, central_contrastive_loss

logger = logging.getLogger("bottle_vision")


@dataclass
class ModelOutput:
    prob_preds: torch.Tensor | dict[str, torch.Tensor]
    losses: LossComponents
    contrast_loss_stats: dict[str, torch.Tensor]
    central_loss_stats: dict[str, torch.Tensor]


@dataclass
class InferenceOutput:
    embed: torch.Tensor
    tag_embed: torch.Tensor
    tag_logits: torch.Tensor
    artist_embed: torch.Tensor
    artist_logits: torch.Tensor
    character_embed: torch.Tensor
    character_logits: torch.Tensor
    quality_score: torch.Tensor


class ContrastivePrototypes(nn.Module):
    """Contrastive prototypes for multi-task metric learning model."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(torch.randn(num_classes, embed_dim), dim=1))


class ContrastiveTemp(nn.Module):
    """Contrastive temperature for multi-task metric learning model."""

    def __init__(self, num_classes: int, temp: float, temp_strategy: Literal["fixed", "task", "class"]):
        super().__init__()

        if temp_strategy == "fixed":
            self.val = temp
        elif temp_strategy == "task":
            self.val = nn.Parameter(torch.tensor(temp))
        elif temp_strategy == "class":
            self.val = nn.Parameter(torch.ones(1, num_classes) * temp)
        else:
            raise ValueError(f"Unknown temperature strategy: {temp_strategy}")


class IllustEmbeddingModel(nn.Module):
    """Multi-task metric learning model architecture for illustrations.

    Learns joint embeddings for tags, artists, and characters.
    """

    def __init__(
        self,
        num_tags: int,
        num_artists: int,
        num_characters: int,
        backbone_variant: str,
        image_size: int,
        tag_embed_dim: int,
        artist_embed_dim: int,
        character_embed_dim: int,
        cls_token: bool,
        reg_tokens: int,
        dropout: float,
        tag_temp: float,
        artist_temp: float,
        character_temp: float,
        tasks: list[str],
        temp_strategy: Literal["fixed", "task", "class"],
    ):
        super().__init__()
        self.tasks = tasks

        # Backbone
        self.backbone = timm.create_model(
            backbone_variant,
            img_size=image_size,
            num_classes=0,
            class_token=cls_token,
            reg_tokens=reg_tokens,
            global_pool="token" if cls_token else "avg",
            fc_norm=False,
            act_layer="gelu_tanh",
        )

        self.hidden_dim = self.backbone.embed_dim
        self.dropout = nn.Dropout(dropout)

        # Projection heads and prototypes
        self.trainable_module_names = []
        if "tag" in tasks:
            self.tag_head = nn.Linear(self.hidden_dim, tag_embed_dim, bias=False)
            nn.init.orthogonal_(self.tag_head.weight)

            self.tag_prototypes = ContrastivePrototypes(num_tags, tag_embed_dim)
            self.tag_temp = ContrastiveTemp(num_tags, tag_temp, temp_strategy)

            self.trainable_module_names += ["tag_head", "tag_prototypes"]
            if temp_strategy != "fixed":
                self.trainable_module_names.append("tag_temp")

        if "character" in tasks:
            self.character_head = nn.Linear(self.hidden_dim, character_embed_dim, bias=False)
            nn.init.orthogonal_(self.character_head.weight)

            self.character_prototypes = ContrastivePrototypes(num_characters, character_embed_dim)
            self.character_temp = ContrastiveTemp(num_characters, character_temp, temp_strategy)

            self.trainable_module_names += ["character_head", "character_prototypes"]
            if temp_strategy != "fixed":
                self.trainable_module_names.append("character_temp")

        if "artist" in tasks:
            self.artist_head = nn.Linear(self.hidden_dim, artist_embed_dim, bias=False)

            self.artist_prototypes = ContrastivePrototypes(num_artists, artist_embed_dim)
            self.artist_temp = ContrastiveTemp(num_artists, artist_temp, temp_strategy)

            self.trainable_module_names += ["artist_head", "artist_prototypes"]
            if temp_strategy != "fixed":
                self.trainable_module_names.append("artist_temp")

        if "quality" in tasks:
            self.quality_head = nn.Linear(self.hidden_dim, 1)
            self.trainable_module_names.append("quality_head")

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: style-related intermediates fusion
        x, intermediates = self.backbone.forward_intermediates(x, indices=[], output_fmt="NLC")

    def _compute_task_output(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task: str,
        contrastive_params: ContrastiveLossParams,
        use_focal_loss: bool = False,
        masks: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute similarities and loss for a specific task."""
        # Select appropriate head and prototypes based on task
        assert task in self.tasks, f"Task {task} not in model tasks: {self.tasks}"
        if task == "tag":
            head = self.tag_head
            prototypes = self.tag_prototypes.weight
            temp = self.tag_temp.val
        elif task == "artist":
            head = self.artist_head
            prototypes = self.artist_prototypes.weight
            temp = self.artist_temp.val
        elif task == "character":
            head = self.character_head
            prototypes = self.character_prototypes.weight
            temp = self.character_temp.val
        else:
            raise ValueError(f"Unknown task: {task}")

        # Compute embeddings and similarities
        embeddings = head(self.dropout(features))
        similarities = F.normalize(embeddings, dim=1) @ F.normalize(prototypes, dim=1).T
        probs = torch.sigmoid(similarities / temp)

        # Compute loss with class weights
        loss, contrast_loss, central_loss = central_contrastive_loss(
            sim=similarities,
            labels=labels,
            temp=temp,
            params=contrastive_params,
            mask=masks,
            prob_preds=probs if use_focal_loss else None,
            class_weights=class_weights,
        )

        return probs, loss, contrast_loss, central_loss

    def forward_task(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        score: torch.Tensor,
        task: str,
        contrastive_params: ContrastiveLossParams,
        use_focal_loss: bool = False,
        masks: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass for a specific task."""
        # Get backbone features
        features = self.backbone(x)

        # Compute task-specific outputs
        prob_preds, task_loss, contrast_loss, central_loss = self._compute_task_output(
            features=features,
            labels=labels,
            task=task,
            contrastive_params=contrastive_params,
            masks=masks,
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
        )
        losses = LossComponents(**{task: task_loss})

        # Compute quality loss
        if "quality" in self.tasks:
            quality_score = self.quality_head(features).squeeze(-1)
            losses.quality = F.mse_loss(quality_score, score)

        return ModelOutput(
            prob_preds=prob_preds,
            losses=losses,
            contrast_loss_stats={task: contrast_loss},
            central_loss_stats={task: central_loss},
        )

    def forward_all_tasks(
        self,
        x: torch.Tensor,
        labels: dict[str, torch.Tensor],
        score: torch.Tensor,
        contrastive_params: dict[str, ContrastiveLossParams],
        use_focal_loss: bool = False,
        masks: dict[str, torch.Tensor] = {},
        class_weights: dict[str, torch.Tensor] = {},
    ) -> ModelOutput:
        """Forward pass for all tasks, used in validation/test."""
        # Get backbone features
        features = self.backbone(x)

        # Compute task-specific outputs
        prob_preds = {}
        contrast_loss_stats = {}
        central_loss_stats = {}
        losses = LossComponents()
        for task, labels in labels.items():
            task_prob_preds, task_loss, contrast_loss, central_loss = self._compute_task_output(
                features=features,
                labels=labels,
                task=task,
                contrastive_params=contrastive_params[task],
                masks=masks.get(task),
                class_weights=class_weights.get(task),
                use_focal_loss=use_focal_loss,
            )
            prob_preds[task] = task_prob_preds
            contrast_loss_stats[task] = contrast_loss
            central_loss_stats[task] = central_loss
            losses.__setattr__(task, task_loss)

        # Compute quality loss
        if "quality" in self.tasks:
            quality_score = self.quality_head(features).squeeze(-1)
            losses.quality = F.mse_loss(quality_score, score)

        return ModelOutput(
            prob_preds=prob_preds,
            losses=losses,
            contrast_loss_stats=contrast_loss_stats,
            central_loss_stats=central_loss_stats,
        )

    def forward(self, x: torch.Tensor) -> InferenceOutput:
        features = self.backbone(x)

        tag_emb = self.tag_head(features) if "tag" in self.tasks else None
        artist_emb = self.artist_head(features) if "artist" in self.tasks else None
        character_emb = self.character_head(features) if "character" in self.tasks else None
        quality_score = self.quality_head(features).squeeze(-1) if "quality" in self.tasks else None

        tag_logits = (
            F.normalize(tag_emb, dim=1) @ F.normalize(self.tag_prototypes, dim=1).T if tag_emb is not None else None
        )
        artist_logits = (
            F.normalize(artist_emb, dim=1) @ F.normalize(self.artist_prototypes, dim=1).T
            if artist_emb is not None
            else None
        )
        character_logits = (
            F.normalize(character_emb, dim=1) @ F.normalize(self.character_prototypes, dim=1).T
            if character_emb is not None
            else None
        )

        return InferenceOutput(
            embed=features,
            tag_embed=tag_emb,
            tag_logits=tag_logits,
            artist_embed=artist_emb,
            artist_logits=artist_logits,
            character_embed=character_emb,
            character_logits=character_logits,
            quality_score=quality_score,
        )

    def load_wd_tagger_weights(self, num_tags: int, num_characters: int):
        """Load weights from WD-Tagger pretrained model."""
        pretrained_model = timm.create_model("hf_hub:SmilingWolf/wd-vit-tagger-v3", pretrained=True)
        # print("source parameters:", [n for n, _ in pretrained_model.named_parameters()])
        # print("target parameters:", [n for n, _ in self.backbone.named_parameters()])

        # patch embedding
        self._copy_weights(self.backbone.patch_embed.proj, pretrained_model.patch_embed.proj)
        logger.info(f"Loaded patch embedding weights: {self.backbone.patch_embed.proj.weight.shape}")

        # positional embedding: ignore prefix tokens, only apply to the last num_patches tokens
        num_patches = self.backbone.patch_embed.num_patches
        self.backbone.pos_embed.data[:, -num_patches:] = pretrained_model.pos_embed.data.to(
            self.backbone.pos_embed.device
        )
        logger.info(
            f"Loaded positional embedding weights: {pretrained_model.pos_embed.shape} -> {self.backbone.pos_embed.shape}"
        )

        # blocks
        for self_block, pretrained_block in zip(self.backbone.blocks, pretrained_model.blocks):
            self._copy_block_weights(self_block, pretrained_block)
        logger.info(f"Loaded weights for {len(self.backbone.blocks)} transformer blocks")

        # norm
        self._copy_weights(self.backbone.norm, pretrained_model.norm)
        logger.info(f"Loaded norm weights: {self.backbone.norm.weight.shape}")

        # head -> (current random head) -> tag and character prototypes
        if num_tags + num_characters == pretrained_model.num_classes:
            if "tag" in self.tasks:
                # extract embeddings from pretrained head
                pretrained_tag_embeddings = pretrained_model.head.weight[:num_tags]
                pretrained_tag_embeddings = pretrained_tag_embeddings.to(self.tag_prototypes.weight.device)
                pretrained_tag_embeddings = F.normalize(pretrained_tag_embeddings, dim=1)
                # project embeddings to tag and character prototypes
                if hasattr(self.tag_head, "weight"):
                    # (num_classes, hidden_dim) @ (class_dim, hidden_dim).T -> (num_classes, class_dim)
                    projected_tag_embeddings = pretrained_tag_embeddings @ self.tag_head.weight.T
                    self.tag_prototypes.weight.data = projected_tag_embeddings
                    logger.info(
                        f"Loaded tag prototypes from head: {self.tag_prototypes.weight.shape}, projected by head"
                    )
                else:
                    self.tag_prototypes.weight.data = pretrained_tag_embeddings
                    logger.info(f"Loaded tag prototypes from head: {self.tag_prototypes.weight.shape}")

            if "character" in self.tasks:
                pretrained_character_embeddings = pretrained_model.head.weight[num_tags:]
                pretrained_character_embeddings = pretrained_character_embeddings.to(
                    self.character_prototypes.weight.device
                )
                pretrained_character_embeddings = F.normalize(pretrained_character_embeddings, dim=1)
                if hasattr(self.character_head, "weight"):
                    projected_character_embeddings = pretrained_character_embeddings @ self.character_head.weight.T
                    self.character_prototypes.weight.data = projected_character_embeddings
                    logger.info(
                        f"Loaded character prototypes from head: {self.character_prototypes.weight.shape}, projected by head"
                    )
                else:
                    self.character_prototypes.weight.data = pretrained_character_embeddings
                    logger.info(f"Loaded character prototypes from head: {self.character_prototypes.weight.shape}")

    def _copy_weights(self, target: nn.Module, source: nn.Module):
        """Copy weights and biases from source to target."""
        assert target.weight.shape == source.weight.shape
        target.weight.data = source.weight.data.to(target.weight.device)
        if hasattr(target, "bias") and target.bias is not None:
            assert target.bias.shape == source.bias.shape
            target.bias.data = source.bias.data.to(target.bias.device)

    def _copy_block_weights(self, target_block: nn.Module, source_block: nn.Module):
        """Copy weights and biases for a transformer block."""
        self._copy_weights(target_block.attn.qkv, source_block.attn.qkv)
        self._copy_weights(target_block.attn.proj, source_block.attn.proj)
        self._copy_weights(target_block.mlp.fc1, source_block.mlp.fc1)
        self._copy_weights(target_block.mlp.fc2, source_block.mlp.fc2)
        self._copy_weights(target_block.norm1, source_block.norm1)
        self._copy_weights(target_block.norm2, source_block.norm2)
