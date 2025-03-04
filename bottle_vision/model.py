import logging
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import ContrastiveLossParams, LossComponents, central_contrastive_loss


@dataclass
class ModelOutput:
    sim_preds: torch.Tensor | dict[str, torch.Tensor]
    losses: LossComponents


class IllustEmbeddingModel(nn.Module):
    """Multi-task metric learning model architecture for illustrations.

    Learns joint embeddings for tags, artists, and characters while maintaining
    orthogonality between different embedding spaces.
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
    ):
        super().__init__()

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

        # Projection heads
        self.hidden_dim = self.backbone.embed_dim
        self.tag_head = nn.Linear(self.hidden_dim, tag_embed_dim)
        self.artist_head = nn.Linear(self.hidden_dim, artist_embed_dim)
        self.character_head = nn.Linear(self.hidden_dim, character_embed_dim)
        self.quality_head = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Learnable prototypes
        self.tag_prototypes = nn.Parameter(torch.randn(num_tags, tag_embed_dim))
        self.character_prototypes = nn.Parameter(torch.randn(num_characters, character_embed_dim))
        self.artist_prototypes = nn.Parameter(torch.randn(num_artists, artist_embed_dim))
        nn.init.xavier_normal_(self.tag_prototypes)
        nn.init.xavier_normal_(self.artist_prototypes)
        nn.init.xavier_normal_(self.character_prototypes)

        self.tag_temp = nn.Parameter(torch.ones(1, num_tags) * tag_temp)
        self.artist_temp = nn.Parameter(torch.ones(1, num_artists) * artist_temp)
        self.character_temp = nn.Parameter(torch.ones(1, num_characters) * character_temp)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        tag_emb = self.dropout(self.tag_head(features))
        artist_emb = self.dropout(self.artist_head(features))
        character_emb = self.dropout(self.character_head(features))
        quality_score = self.quality_head(features).squeeze(-1)
        return tag_emb, artist_emb, character_emb, quality_score

    def forward_task(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        score: torch.Tensor,
        task: str,
        contrastive_params: ContrastiveLossParams,
    ) -> ModelOutput:
        """Forward pass for a specific task."""
        # Get backbone features
        features = self.backbone(x)

        # Compute task-specific outputs
        sim_preds, task_loss = self._compute_task_output(
            features=features,
            labels=labels,
            task=task,
            contrastive_params=contrastive_params,
        )

        # Compute quality loss
        quality_score = self.quality_head(features).squeeze(-1)
        quality_loss = F.mse_loss(quality_score, score)

        # Compute orthogonality losses
        ortho_loss = self._compute_ortho_loss(task)

        losses = LossComponents(quality=quality_loss, ortho=ortho_loss, **{task: task_loss})
        return ModelOutput(sim_preds=sim_preds, losses=losses)

    def _compute_task_output(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task: str,
        contrastive_params: ContrastiveLossParams,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute similarities and loss for a specific task."""
        # Select appropriate head and prototypes based on task
        if task == "tag":
            head = self.tag_head
            prototypes = self.tag_prototypes
            temp = self.tag_temp
        elif task == "artist":
            head = self.artist_head
            prototypes = self.artist_prototypes
            temp = self.artist_temp
        elif task == "character":
            head = self.character_head
            prototypes = self.character_prototypes
            temp = self.character_temp
        else:
            raise ValueError(f"Unknown task: {task}")

        # Compute embeddings and similarities
        embeddings = self.dropout(head(features))
        similarities = F.normalize(embeddings, dim=1) @ F.normalize(prototypes, dim=1).T

        # Compute loss with class weights
        loss = central_contrastive_loss(
            sim=similarities,
            labels=labels,
            temp=temp,
            params=contrastive_params,
        )

        return torch.sigmoid(similarities), loss

    def forward_all_tasks(
        self,
        x: torch.Tensor,
        labels: dict[str, torch.Tensor],
        score: torch.Tensor,
        contrastive_params: dict[str, ContrastiveLossParams],
    ) -> ModelOutput:
        """Forward pass for all tasks, used in validation/test."""
        # Get backbone features
        features = self.backbone(x)

        # Compute task-specific outputs
        sim_preds = {}
        losses = LossComponents()
        for task, labels in labels.items():
            task_sim_preds, task_loss = self._compute_task_output(
                features=features,
                labels=labels,
                task=task,
                contrastive_params=contrastive_params[task],
            )
            sim_preds[task] = task_sim_preds
            losses.__setattr__(task, task_loss)

        # Compute quality loss
        quality_score = self.quality_head(features).squeeze(-1)
        losses.quality = F.mse_loss(quality_score, score)

        # Compute orthogonality losses
        losses.ortho = self._compute_ortho_loss("artist")

        return ModelOutput(sim_preds=sim_preds, losses=losses)

    def _compute_ortho_loss(self, task: str) -> torch.Tensor:
        """Compute orthogonality loss between embedding spaces based on task."""
        if task == "tag":
            # Tag embeddings should be orthogonal to artist embeddings
            return torch.norm(self.tag_head.weight @ self.artist_head.weight.T, p="fro") ** 2

        elif task == "artist":
            # Artist embeddings should be orthogonal to tag and character embeddings
            tag_ortho = torch.norm(self.artist_head.weight @ self.tag_head.weight.T, p="fro") ** 2
            char_ortho = torch.norm(self.artist_head.weight @ self.character_head.weight.T, p="fro") ** 2
            return tag_ortho + char_ortho

        elif task == "character":
            # Character embeddings should be orthogonal to artist embeddings
            return torch.norm(self.character_head.weight @ self.artist_head.weight.T, p="fro") ** 2

        raise ValueError(f"Unknown task: {task}")

    def load_wd_tagger_weights(self, num_tags: int, num_artists: int):
        """Load weights from WD-Tagger pretrained model."""
        pretrained_model = timm.create_model("hf_hub:SmilingWolf/wd-vit-tagger-v3", pretrained=True)

        # patch embedding
        self._copy_weights(self.backbone.patch_embed.proj, pretrained_model.patch_embed.proj)
        logging.info(f"Loaded patch embedding weights: {self.backbone.patch_embed.proj.weight.shape}")

        # positional embedding: ignore prefix tokens, only apply to the last num_patches tokens
        num_patches = self.backbone.patch_embed.num_patches
        self.backbone.pos_embed.data[0, -num_patches:] = pretrained_model.pos_embed.data
        logging.info(
            f"Loaded positional embedding weights: {pretrained_model.pos_embed.shape} -> {self.backbone.pos_embed.shape}"
        )

        # blocks
        for self_block, pretrained_block in zip(self.backbone.blocks, pretrained_model.blocks):
            self._copy_block_weights(self_block, pretrained_block)
        logging.info(f"Loaded weights for {len(self.backbone.blocks)} transformer blocks")

        # norm
        self._copy_weights(self.backbone.norm, pretrained_model.norm)
        logging.info(f"Loaded norm weights: {self.backbone.norm.weight.shape}")

        # head -> (current random head) -> tag and character prototypes
        if num_tags + num_characters == pretrained_model.num_classes:
            # extract embeddings from pretrained head
            pretrained_tag_embeddings = pretrained_model.head.weight[:num_tags].to(self.tag_head.weight.device)
            pretrained_character_embeddings = pretrained_model.head.weight[num_tags:].to(
                self.character_head.weight.device
            )

            # project embeddings to tag and character prototypes
            # (num_classes, hidden_dim) @ (class_dim, hidden_dim).T -> (num_classes, class_dim)
            self.tag_prototypes.data = pretrained_tag_embeddings @ self.tag_head.weight.T
            self.character_prototypes.data = pretrained_character_embeddings @ self.character_head.weight.T
            logger.info(f"Loaded tag prototypes from head: {self.tag_prototypes.shape}")
            logger.info(f"Loaded character prototypes from head: {self.character_prototypes.shape}")

    def _copy_weights(self, target: nn.Module, source: nn.Module):
        """Copy weights and biases from source to target."""
        assert target.weight.shape == source.weight.shape
        target.weight.data = source.weight.data
        if hasattr(target, "bias") and target.bias is not None:
            assert target.bias.shape == source.bias.shape
            target.bias.data = source.bias.data

    def _copy_block_weights(self, target_block: nn.Module, source_block: nn.Module):
        """Copy weights and biases for a transformer block."""
        self._copy_weights(target_block.attn.qkv, source_block.attn.qkv)
        self._copy_weights(target_block.attn.proj, source_block.attn.proj)
        self._copy_weights(target_block.mlp.fc1, source_block.mlp.fc1)
        self._copy_weights(target_block.mlp.fc2, source_block.mlp.fc2)
        self._copy_weights(target_block.norm1, source_block.norm1)
        self._copy_weights(target_block.norm2, source_block.norm2)

    def freeze_loaded_weights(self):
        """Freeze all weights that have been loaded in the load_wd_tagger_weights function."""
        # Freeze patch embedding weights
        self.backbone.patch_embed.proj.weight.requires_grad = False
        self.backbone.patch_embed.proj.bias.requires_grad = False
        logging.info("Frozen patch embedding weights")

        # No way to freeze partial positional embedding weights

        # Freeze weights for each transformer block
        for block in self.backbone.blocks:
            for param in block.parameters():
                param.requires_grad = False
        logging.info(f"Froze weights for {len(self.backbone.blocks)} transformer blocks")

        # Freeze norm weights
        self.backbone.norm.weight.requires_grad = False
        self.backbone.norm.bias.requires_grad = False
        logging.info("Frozen norm weights")

        # No need to freeze tag and artist prototypes, since heads are learnable

    def unfreeze_deeper_layers(self, num_layers: int):
        """Unfreeze the last num_layers transformer blocks."""
        for block in self.backbone.blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        logging.info(f"Unfroze the last {num_layers} transformer blocks")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logging.info("Unfroze all parameters")
