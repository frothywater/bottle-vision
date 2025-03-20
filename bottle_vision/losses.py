from dataclasses import dataclass
from typing import Optional, TypedDict, Union

import torch


class ContrastiveLossConfig(TypedDict, total=False):
    temp: float
    margin: float
    central_weight: float
    min_temp: float
    max_temp: float
    min_margin: float
    max_margin: float

    @staticmethod
    def default() -> "ContrastiveLossConfig":
        return ContrastiveLossConfig(
            temp=0.1,
            margin=0.2,
            central_weight=1.5,
            min_temp=None,
            max_temp=None,
            min_margin=None,
            max_margin=None,
        )

    @staticmethod
    def create(**kwargs) -> "ContrastiveLossConfig":
        _kwargs = ContrastiveLossConfig.default()
        _kwargs.update(kwargs)
        return ContrastiveLossConfig(**_kwargs)


@dataclass
class ContrastiveLossParams:
    margin: float | torch.Tensor
    central_weight: float

    @staticmethod
    def from_config(config: ContrastiveLossConfig, margin: Optional[torch.Tensor] = None) -> "ContrastiveLossParams":
        return ContrastiveLossParams(
            margin=margin if margin is not None else config["margin"],
            central_weight=config["central_weight"],
        )


class LossWeights(TypedDict, total=False):
    tag: float
    artist: float
    character: float
    quality: float

    @staticmethod
    def default() -> "LossWeights":
        return LossWeights(
            tag=1.0,
            artist=1.0,
            character=1.0,
            quality=1.0,
        )

    @staticmethod
    def create(**kwargs) -> "LossWeights":
        _kwargs = LossWeights.default()
        _kwargs.update(kwargs)
        return LossWeights(**_kwargs)


@dataclass
class LossComponents:
    tag: float = 0.0
    artist: float = 0.0
    character: float = 0.0
    quality: float = 0.0

    def weighted_sum(self, weights: LossWeights) -> torch.Tensor:
        return (
            weights["tag"] * self.tag
            + weights["artist"] * self.artist
            + weights["character"] * self.character
            + weights["quality"] * self.quality
        )


def central_contrastive_loss(
    sim: torch.Tensor,
    labels: torch.Tensor,
    temp: Union[float, torch.Tensor],
    params: ContrastiveLossParams,
    eps: float = 1e-6,
    max_exp_value: float = 30.0,
    mask: torch.Tensor = None,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor = None,
    prob_preds: torch.Tensor = None,
) -> torch.Tensor:
    """Central contrastive loss adapted for multi-label supervised case with class weights.

    Args:
        sim: Similarity matrix of shape (N, K)
        labels: Binary label matrix of shape (N, K)
        temp: Temperature parameter
        params: Contrastive loss parameters
        eps: Small value to avoid numerical instability
        mask: Bool mask for positive labels of shape (N,)

    Returns:
        Loss value as a scalar tensor

    References:
        - [Center Contrastive Loss for Metric Learning](https://arxiv.org/abs/2308.00458)
        - [Class Prototypes Based Contrastive Learning for Classifying Multi-Label and Fine-Grained Educational Videos](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Class_Prototypes_Based_Contrastive_Learning_for_Classifying_Multi-Label_and_Fine-Grained_CVPR_2023_paper.html)
        - [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)

    Note:
        Too small eps (1e-8) for fp16 training may cause NaN loss values.
    """
    if mask is not None:
        # mask out samples with no positive labels
        # if all labels are negative, return zero loss
        if not mask.any():
            return 0.0, 0.0, 0.0
        sim = sim[mask]
        labels = labels[mask]

    # compute a per-sample shift to improve stability (log-sum-exp trick)
    sim_div = sim / temp
    shift = sim_div.max(dim=1, keepdim=True).values

    # negative samples, broadcasted
    negatives = torch.exp(torch.clamp(sim_div - shift, max=max_exp_value)) * (1 - labels)
    negatives = negatives.sum(dim=1, keepdim=True)

    # positive samples
    positives = torch.exp(torch.clamp(sim_div - shift - params.margin / temp, max=max_exp_value)) * labels
    # focal modulation
    if prob_preds is not None:
        positives *= (1 - prob_preds) ** focal_gamma

    # log ratio, mask out negative samples
    ratio = positives / (positives + negatives + eps)
    contrast_loss = -torch.log(ratio + eps) * labels

    # central loss: minimize distance between positive samples and their centroids
    if params.central_weight > 0:
        central_loss = -2 * sim * labels
    else:
        central_loss = 0.0

    loss = contrast_loss + params.central_weight * central_loss
    # return each part of the raw loss for monitoring
    label_sum = labels.sum(dim=1) + eps
    contrast_loss = (contrast_loss.sum(dim=1) / label_sum).mean()
    if params.central_weight > 0:
        central_loss = (central_loss.sum(dim=1) / label_sum).mean()

    # reweight positive samples based on class weights
    if class_weights is not None:
        positive_mask = labels.round().bool()
        loss = torch.where(positive_mask, loss * class_weights, loss)
        labels = torch.where(positive_mask, labels * class_weights, labels)

    # average over positive labels
    loss = loss.sum(dim=1) / (labels.sum(dim=1) + eps)
    # average over samples
    loss = loss.mean()

    return loss, contrast_loss, central_loss
