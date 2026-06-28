import torch
import torch.nn as nn


class Conv2dResidualAdapter(nn.Module):
    """
    Frozen Conv2d plus a trainable bottleneck residual adapter.

        y = conv(x) + alpha * up(act(down(x)))

    This wrapper is intended for stride-1 decoder convolutions whose output
    spatial shape matches the input spatial shape.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        reduction: int = 16,
        activation: str = "GELU",
        alpha_init: float = 1.0,
        train_alpha: bool = True,
    ):
        super().__init__()

        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"Conv2dResidualAdapter expects nn.Conv2d, got {type(conv)}")
        if conv.stride != (1, 1):
            raise ValueError("Conv2dResidualAdapter only supports stride-1 Conv2d")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")

        self.original_conv = conv
        for param in self.original_conv.parameters():
            param.requires_grad = False

        hidden_channels = max(1, conv.out_channels // reduction)
        self.decoder_adapter_down = nn.Conv2d(
            conv.in_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.decoder_adapter_activation = self._build_activation(activation)
        self.decoder_adapter_up = nn.Conv2d(
            hidden_channels,
            conv.out_channels,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(self.decoder_adapter_up.weight)
        nn.init.zeros_(self.decoder_adapter_up.bias)

        if train_alpha:
            self.decoder_adapter_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer(
                "decoder_adapter_alpha",
                torch.tensor(float(alpha_init)),
                persistent=True,
            )

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "gelu":
            return nn.GELU()
        if name == "relu":
            return nn.ReLU()
        if name in {"silu", "swish"}:
            return nn.SiLU()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(
            f"Unsupported decoder adapter activation: {name}. "
            "Supported: GELU, ReLU, SiLU, Tanh."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original_conv(x)
        adapter = self.decoder_adapter_down(x)
        adapter = self.decoder_adapter_activation(adapter)
        adapter = self.decoder_adapter_up(adapter)
        return base + self.decoder_adapter_alpha * adapter
