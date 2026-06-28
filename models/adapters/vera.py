import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VeRAQKV(nn.Module):
    """
    VeRA-style wrapper for a qkv projection.

    The pretrained qkv projection is frozen. Each selected q/k/v component gets
    a frozen random low-rank basis and trainable scaling vectors. Random bases
    are registered as non-persistent buffers so checkpoints only store the small
    trainable VeRA parameters and can be reconstructed deterministically from
    config before strict checkpoint loading.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int,
        alpha: float,
        targets: Iterable[str] = ("q", "v"),
        dropout: float = 0.0,
        train_alpha: bool = True,
        down_weight: Optional[torch.Tensor] = None,
        up_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if not isinstance(linear_layer, nn.Linear):
            raise TypeError(f"VeRAQKV expects nn.Linear, got {type(linear_layer)}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.linear_layer = linear_layer
        self.in_dim = linear_layer.in_features
        self.out_dim = linear_layer.out_features

        if self.out_dim != 3 * self.in_dim:
            raise ValueError(
                "VeRAQKV expects qkv out_features = 3 * in_features. "
                f"Got in_features={self.in_dim}, out_features={self.out_dim}."
            )

        self.rank = rank
        self.targets: Tuple[str, ...] = tuple(targets)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        allowed = {"q", "k", "v"}
        for target in self.targets:
            if target not in allowed:
                raise ValueError(f"Unknown VeRA target '{target}'. Allowed: {allowed}")

        for param in self.linear_layer.parameters():
            param.requires_grad = False

        if down_weight is None:
            down_weight = torch.empty(rank, self.in_dim)
            nn.init.kaiming_uniform_(down_weight, a=math.sqrt(5))
        if up_weight is None:
            up_weight = torch.empty(self.in_dim, rank)
            nn.init.kaiming_uniform_(up_weight, a=math.sqrt(5))

        if tuple(down_weight.shape) != (rank, self.in_dim):
            raise ValueError(f"down_weight must have shape {(rank, self.in_dim)}")
        if tuple(up_weight.shape) != (self.in_dim, rank):
            raise ValueError(f"up_weight must have shape {(self.in_dim, rank)}")

        self.register_buffer("vera_down_weight", down_weight.clone(), persistent=False)
        self.register_buffer("vera_up_weight", up_weight.clone(), persistent=False)

        if train_alpha:
            self.vera_alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("vera_alpha", torch.tensor(float(alpha)), persistent=False)

        self._make_target_parameters("q")
        self._make_target_parameters("k")
        self._make_target_parameters("v")

    def _make_target_parameters(self, target: str) -> None:
        if target in self.targets:
            setattr(
                self,
                f"vera_adapter_{target}_scale_down",
                nn.Parameter(torch.ones(self.rank)),
            )
            # Zero-init output scaling so the wrapper is an exact no-op at step 0.
            setattr(
                self,
                f"vera_adapter_{target}_scale_up",
                nn.Parameter(torch.zeros(self.in_dim)),
            )
        else:
            self.register_parameter(f"vera_adapter_{target}_scale_down", None)
            self.register_parameter(f"vera_adapter_{target}_scale_up", None)

    def _adapter_forward(self, x: torch.Tensor, target: str) -> torch.Tensor:
        scale_down = getattr(self, f"vera_adapter_{target}_scale_down")
        scale_up = getattr(self, f"vera_adapter_{target}_scale_up")
        if scale_down is None or scale_up is None:
            return torch.zeros_like(x)

        hidden = F.linear(self.dropout(x), self.vera_down_weight)
        hidden = hidden * scale_down
        out = F.linear(hidden, self.vera_up_weight)
        out = out * scale_up
        return out * (self.vera_alpha / float(self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear_layer(x)
        delta_q = self._adapter_forward(x, "q")
        delta_k = self._adapter_forward(x, "k")
        delta_v = self._adapter_forward(x, "v")
        delta = torch.cat([delta_q, delta_k, delta_v], dim=-1)
        return base + delta
