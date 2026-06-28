import torch
import torch.nn as nn
# adaptformer insists con MLP Block of the transformer

class AdaptFormer(nn.Module):
    """
    AdaptFormer wrapper for a Transformer MLP block.

    It preserves the original MLP path and adds a trainable bottleneck adapter:

        output = original_mlp(norm(x)) + alpha * adapter(x)

    The up-projection is initialized to zero, so at initialization the adapter
    does not modify the pretrained model.
    """

    def __init__(
        self,
        layer_norm: nn.Module,
        mlp_block: nn.Module,
        adapter_activation: str = "GELU",
        reduction_factor: int = 16,
    ):
        super().__init__()

        if reduction_factor <= 0:
            raise ValueError(f"reduction_factor must be positive, got {reduction_factor}")

        self.layer_norm = layer_norm
        self.mlp_block = mlp_block
        self.adapter_alpha = nn.Parameter(torch.ones(1))

        input_dim = self._infer_input_dim(mlp_block)
        hidden_dim = max(1, input_dim // reduction_factor)

        self.adapter_downsample = nn.Linear(input_dim, hidden_dim)
        self.adapter_activation = self._build_activation(adapter_activation)
        self.adapter_upsample = nn.Linear(hidden_dim, input_dim)

        # Zero-init the up projection so the adapter starts as a no-op.
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)

        # Freeze the original pretrained MLP and its LayerNorm.
        for param in self.mlp_block.parameters():
            param.requires_grad = False

        for param in self.layer_norm.parameters():
            param.requires_grad = False

    @staticmethod
    def _infer_input_dim(mlp_block: nn.Module) -> int:
        """
        Infer the input dimension of common ViT/SAM MLP blocks.
        Supports both timm-like and SAM-like naming conventions.
        """
        if hasattr(mlp_block, "fc1"):
            return mlp_block.fc1.in_features

        if hasattr(mlp_block, "lin1"):
            return mlp_block.lin1.in_features

        if hasattr(mlp_block, "layers") and len(mlp_block.layers) > 0:
            first = mlp_block.layers[0]
            if isinstance(first, nn.Linear):
                return first.in_features

        if isinstance(mlp_block, nn.Sequential):
            for layer in mlp_block:
                if isinstance(layer, nn.Linear):
                    return layer.in_features

        raise AttributeError(
            "Could not infer MLP input dimension. Expected one of: "
            "mlp_block.fc1, mlp_block.lin1, mlp_block.layers[0], "
            "or a Sequential block containing a Linear layer."
        )

    @staticmethod
    def _build_activation(adapter_activation: str) -> nn.Module:
        name = adapter_activation.lower()

        if name == "gelu":
            return nn.GELU()
        if name == "relu":
            return nn.ReLU()
        if name == "silu" or name == "swish":
            return nn.SiLU()
        if name == "tanh":
            return nn.Tanh()

        raise ValueError(
            f"Unsupported adapter activation: {adapter_activation}. "
            "Supported: GELU, ReLU, SiLU, Tanh."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_x = self.mlp_block(self.layer_norm(x))

        adapter_x = self.adapter_downsample(x)
        adapter_x = self.adapter_activation(adapter_x)
        adapter_x = self.adapter_upsample(adapter_x)

        return main_x + self.adapter_alpha * adapter_x