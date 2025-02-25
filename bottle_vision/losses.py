from dataclasses import dataclass
from typing import TypedDict, Union

import torch


class ContrastiveLossConfig(TypedDict):
    initial_temp: float = 0.1
    margin: float = 0.2
    central_weight: float = 1.5


@dataclass
class ContrastiveLossParams:
    margin: float
    central_weight: float

    @staticmethod
    def from_config(config: ContrastiveLossConfig) -> "ContrastiveLossParams":
        return ContrastiveLossParams(
            margin=config["margin"],
            central_weight=config["central_weight"],
        )


class LossWeights(TypedDict):
    tag: float = 1.0
    artist: float = 1.0
    character: float = 1.0
    quality: float = 1.0
    ortho: float = 1.0


@dataclass
class LossComponents:
    tag: float = 0.0
    artist: float = 0.0
    character: float = 0.0
    quality: float = 0.0
    ortho: float = 0.0

    def weighted_sum(self, weights: LossWeights) -> torch.Tensor:
        return (
            weights["tag"] * self.tag
            + weights["artist"] * self.artist
            + weights["character"] * self.character
            + weights["quality"] * self.quality
            + weights["ortho"] * self.ortho
        )


def central_contrastive_loss(
    sim: torch.Tensor,
    labels: torch.Tensor,
    temp: Union[float, torch.Tensor],
    params: ContrastiveLossParams,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Central contrastive loss adapted for multi-label supervised case with class weights.

    Args:
        sim: Similarity matrix of shape (N, K)
        labels: Binary label matrix of shape (N, K)
        temp: Temperature parameter
        params: Contrastive loss parameters
        eps: Small value to avoid numerical instability

    Returns:
        Loss value as a scalar tensor

    Note:
        - Make sure each sample has at least one positive label.

    References:
        - [Center Contrastive Loss for Metric Learning](https://arxiv.org/abs/2308.00458)
        - [Class Prototypes Based Contrastive Learning for Classifying Multi-Label and Fine-Grained Educational Videos](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Class_Prototypes_Based_Contrastive_Learning_for_Classifying_Multi-Label_and_Fine-Grained_CVPR_2023_paper.html)
        - [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
    """
    # negative samples, broadcasted
    negatives = torch.exp(sim / temp) * (1 - labels)
    negatives = negatives.sum(dim=1, keepdim=True)

    # positive samples
    positives = torch.exp((sim - params.margin) / temp) * labels

    # log ratio, mask out negative samples
    ratio = positives / (positives + negatives + eps)
    log_ratio = torch.log(ratio + eps) * labels

    # central loss: minimize distance between positive samples and their centroids
    central_loss = 2 * sim * labels

    loss = -(log_ratio + params.central_weight * central_loss)
    # average over positive labels
    loss = loss.sum(dim=1) / (labels.sum(dim=1) + eps)
    return loss.sum()
