import math
import re

import torch.nn as nn


class LoraLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bias: bool = True,
        lora_dropout: float = 0.0,
    ):
        """
        LoraLinear wraps a nn.Linear layer (as base_layer) with LoRA adapters.
        The LoRA branch is: dropout -> lora_A -> lora_B, scaled by alpha/sqrt(rank).
        The base_layer weight is frozen.
        """
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(rank)

        # According to the reference, lora_A is initialized normally,
        # and lora_B is initialized to zero.
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_B.weight.data.zero_()

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + lora_out * self.scaling


def apply_lora(
    model: nn.Module,
    r: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    bias: bool = True,
    target_modules: list = ["qkv", "fc"],
    layers_pattern: str = "blocks",
    layers_to_transform: list = [8, 9, 10, 11],
    modules_to_save: list = ["norm"],
):
    if isinstance(bias, str):
        bias = bias != "none"

    def matches_any(name, patterns):
        return any(re.search(pat, name) for pat in patterns)

    def replace_module(parent, attr_name, module):
        if not isinstance(module, nn.Linear):
            return
        has_bias = module.bias is not None
        new_module = LoraLinear(
            module.in_features, module.out_features, bias=has_bias, rank=r, alpha=lora_alpha, lora_dropout=lora_dropout
        )
        # Copy the weight and bias (if exists) from the original module.
        new_module.base_layer.weight.data.copy_(module.weight.data)
        if has_bias:
            new_module.base_layer.bias.data.copy_(module.bias.data)
        setattr(parent, attr_name, new_module)

    # Find the layers module.
    layers_module = None
    for name, module in model.named_modules():
        if matches_any(name, [layers_pattern]) and (
            isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential)
        ):
            layers_module = module
            break
    if layers_module is None:
        raise ValueError(f"Could not find layers module with pattern {layers_pattern}")

    for idx, submodule in enumerate(layers_module):
        if idx not in layers_to_transform:
            continue
        # For each named child in this block, check for target module names.
        for name, module in list(submodule.named_children()):
            if matches_any(name, target_modules):
                replace_module(submodule, name, module)

        # Also, for deeper submodules (if nested inside named children), do a recursive walk.
        for child_name, child_module in submodule.named_modules():
            # Avoid re-replacing already replaced modules (they are LoraLinear now).
            if isinstance(child_module, nn.Linear) and matches_any(child_name, target_modules):
                # Find its parent and attribute name by splitting the full name.
                path = child_name.split(".")
                parent_mod = submodule
                for part in path[:-1]:
                    parent_mod = getattr(parent_mod, part)
                attr = path[-1]
                replace_module(parent_mod, attr, child_module)

    # Freeze all parameters by default.
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters and base_layer.bias (if bias is True) in LoraLinear modules.
    for module in model.modules():
        if isinstance(module, LoraLinear):
            # LoRA parameters remain trainable.
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
            # Ensure base_layer.bias remains trainable if bias is True.
            if bias and module.base_layer.bias is not None:
                module.base_layer.bias.requires_grad = True

    # Unfreeze any parameters matching modules_to_save.
    for name, param in model.named_parameters():
        if matches_any(name, modules_to_save):
            param.requires_grad = True

    # Collect names of trainable parameters.
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    return model, trainable_params
