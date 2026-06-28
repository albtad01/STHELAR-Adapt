# lora2 is used, not lora
import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn

# scaling alpha/rank
# frozen original layer 
# initializes up-projection to zero, so at the beginning model = pretrained
# possible to decide whether to adapt q,k,v (default q,v)

class LoRAQKV(nn.Module):
    """
    LoRA wrapper for a qkv projection layer.

    The wrapped layer is expected to map:
        (..., D) -> (..., 3D)

    The output convention is assumed to be:
        [Q, K, V]

    By default, LoRA is applied only to Q and V, while K is left unchanged.
    This is a common and stable choice for attention adaptation.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int,
        alpha: float,
        targets: Iterable[str] = ("q", "v"),
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(linear_layer, nn.Linear):
            raise TypeError(
                f"LoRAQKV expects nn.Linear, got {type(linear_layer)}"
            )

        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.linear_layer = linear_layer
        self.in_dim = linear_layer.in_features
        self.out_dim = linear_layer.out_features

        if self.out_dim != 3 * self.in_dim:
            raise ValueError(
                "LoRAQKV expects a qkv projection with out_features = 3 * in_features. "
                f"Got in_features={self.in_dim}, out_features={self.out_dim}."
            )

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.targets: Tuple[str, ...] = tuple(targets)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        allowed = {"q", "k", "v"}
        for target in self.targets:
            if target not in allowed:
                raise ValueError(
                    f"Unknown LoRA target '{target}'. Allowed targets are {allowed}."
                )

        # Freeze the original pretrained qkv projection.
        for param in self.linear_layer.parameters():
            param.requires_grad = False

        # Adapter modules. Names contain "adapter" so they remain trainable
        # with your current freeze logic.
        if "q" in self.targets:
            self.adapter_q_down = nn.Linear(self.in_dim, rank, bias=False)
            self.adapter_q_up = nn.Linear(rank, self.in_dim, bias=False)
        else:
            self.adapter_q_down = None
            self.adapter_q_up = None

        if "k" in self.targets:
            self.adapter_k_down = nn.Linear(self.in_dim, rank, bias=False)
            self.adapter_k_up = nn.Linear(rank, self.in_dim, bias=False)
        else:
            self.adapter_k_down = None
            self.adapter_k_up = None

        if "v" in self.targets:
            self.adapter_v_down = nn.Linear(self.in_dim, rank, bias=False)
            self.adapter_v_up = nn.Linear(rank, self.in_dim, bias=False)
        else:
            self.adapter_v_down = None
            self.adapter_v_up = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize down projections randomly and up projections to zero.

        Because the up projections are zero-initialized, the LoRA update is
        initially zero. Therefore, at step 0 the wrapped layer behaves exactly
        like the pretrained model.
        """
        for down, up in [
            (self.adapter_q_down, self.adapter_q_up),
            (self.adapter_k_down, self.adapter_k_up),
            (self.adapter_v_down, self.adapter_v_up),
        ]:
            if down is not None:
                nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
                nn.init.zeros_(up.weight)

    def _adapter_forward(self, x: torch.Tensor, down: nn.Linear, up: nn.Linear) -> torch.Tensor:
        return up(down(self.dropout(x))) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear_layer(x)

        zero = torch.zeros_like(x)

        if self.adapter_q_down is not None:
            delta_q = self._adapter_forward(x, self.adapter_q_down, self.adapter_q_up)
        else:
            delta_q = zero

        if self.adapter_k_down is not None:
            delta_k = self._adapter_forward(x, self.adapter_k_down, self.adapter_k_up)
        else:
            delta_k = zero

        if self.adapter_v_down is not None:
            delta_v = self._adapter_forward(x, self.adapter_v_down, self.adapter_v_up)
        else:
            delta_v = zero

        delta = torch.cat([delta_q, delta_k, delta_v], dim=-1)
        return base + delta


# Optional alias, useful if you want to replace the old LoRA class transparently.
LoRA = LoRAQKV