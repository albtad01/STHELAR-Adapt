from collections import defaultdict

import torch
import torch.nn as nn

from .lora2 import LoRAQKV
from .plora import PLoRA
from .bottleneck import BottleNeck
from .adaptformer import AdaptFormer
from .vera import VeRAQKV
from .decoder_conv_adapter import Conv2dResidualAdapter


SHARED_DECODER_PREFIXES = ("decoder0", "decoder1", "decoder2", "decoder3")
BRANCH_DECODER_PREFIXES = (
    "nuclei_binary_map_decoder",
    "hv_map_decoder",
    "nuclei_type_maps_decoder",
)
DECODER_TRAIN_SCOPES = {
    "all",
    "none",
    "nt_only",
    "heads_only",
    "last_stage",
    "nt_header1_np_hv_heads",
    "nt_header_np_hv_heads",
    "np_hv_heads_nt_all",
    "conv_adapters",
}


def insert_lora2(model, rank, alpha, targets=("q", "v"), dropout=0.0):
    """
    Insert cleaned LoRAQKV into the qkv projection of each encoder block.
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.attn.qkv = LoRAQKV(
            linear_layer=encoder_block.attn.qkv,
            rank=rank,
            alpha=alpha,
            targets=targets,
            dropout=dropout,
        )


def insert_lora(model, rank, alpha, targets=("q", "v"), dropout=0.0):
    """
    Backward-compatible alias for insert_lora2.
    """
    insert_lora2(
        model=model,
        rank=rank,
        alpha=alpha,
        targets=targets,
        dropout=dropout,
    )


def insert_vera(
    model,
    rank,
    alpha,
    targets=("q", "v"),
    dropout=0.0,
    shared_matrices=True,
    train_alpha=True,
    seed=42,
):
    """
    Insert VeRA-style QKV adapters into each encoder block.

    Random low-rank matrices are deterministic and non-persistent. When
    shared_matrices=True, all blocks use the same random basis tensors.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    shared_down = None
    shared_up = None
    if shared_matrices:
        in_dim = model.encoder.blocks[0].attn.qkv.in_features
        shared_down = torch.empty(rank, in_dim)
        shared_up = torch.empty(in_dim, rank)
        nn.init.kaiming_uniform_(shared_down, a=5 ** 0.5, generator=generator)
        nn.init.kaiming_uniform_(shared_up, a=5 ** 0.5, generator=generator)

    for encoder_block in model.encoder.blocks:
        down = shared_down
        up = shared_up
        if not shared_matrices:
            in_dim = encoder_block.attn.qkv.in_features
            down = torch.empty(rank, in_dim)
            up = torch.empty(in_dim, rank)
            nn.init.kaiming_uniform_(down, a=5 ** 0.5, generator=generator)
            nn.init.kaiming_uniform_(up, a=5 ** 0.5, generator=generator)

        encoder_block.attn.qkv = VeRAQKV(
            linear_layer=encoder_block.attn.qkv,
            rank=rank,
            alpha=alpha,
            targets=targets,
            dropout=dropout,
            train_alpha=train_alpha,
            down_weight=down,
            up_weight=up,
        )


def insert_plora(model, rank, alpha):
    """
    Insert PLoRA into the qkv projection of each encoder block.
    Kept for compatibility, although PLoRA is currently not the main path.
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.attn.qkv = PLoRA(
            linear_layer=encoder_block.attn.qkv,
            rank=rank,
            alpha=alpha,
        )


def insert_adaptformer(model, activation, reduction):
    """
    Insert AdaptFormer into the MLP block of each encoder block.
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.mlp = AdaptFormer(
            layer_norm=encoder_block.norm2,
            mlp_block=encoder_block.mlp,
            adapter_activation=activation,
            reduction_factor=reduction,
        )
        encoder_block.norm2 = torch.nn.Identity()


def _is_final_decoder_head_module_name(name: str) -> bool:
    return any(
        name == f"{prefix}.decoder0_header.2" for prefix in BRANCH_DECODER_PREFIXES
    )


def _insert_decoder_adapters_recursive(
    module: nn.Module,
    prefix: str,
    reduction: int,
    activation: str,
    alpha_init: float,
    train_alpha: bool,
    inserted: list[str],
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, Conv2dResidualAdapter):
            continue
        if isinstance(child, nn.Conv2d):
            if _is_final_decoder_head_module_name(full_name):
                continue
            if child.stride != (1, 1):
                continue
            setattr(
                module,
                child_name,
                Conv2dResidualAdapter(
                    child,
                    reduction=reduction,
                    activation=activation,
                    alpha_init=alpha_init,
                    train_alpha=train_alpha,
                ),
            )
            inserted.append(full_name)
        else:
            _insert_decoder_adapters_recursive(
                child,
                full_name,
                reduction,
                activation,
                alpha_init,
                train_alpha,
                inserted,
            )


def insert_decoder_conv_adapters(
    model,
    reduction=16,
    activation="GELU",
    alpha_init=1.0,
    train_alpha=True,
):
    """
    Wrap stride-1 Conv2d modules in CellViT decoder paths with residual
    bottleneck adapters. Final output heads are left unwrapped and can be
    trained directly by the conv_adapters decoder scope.
    """
    inserted: list[str] = []
    roots = list(SHARED_DECODER_PREFIXES) + list(BRANCH_DECODER_PREFIXES)
    for root_name in roots:
        if hasattr(model, root_name):
            _insert_decoder_adapters_recursive(
                getattr(model, root_name),
                root_name,
                reduction,
                activation,
                alpha_init,
                train_alpha,
                inserted,
            )
    return inserted


def insert_bottleneck(model, activation, reduction):
    """
    Insert bottleneck adapters after attention and MLP blocks.
    Kept for compatibility.
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.attn = torch.nn.Sequential(
            encoder_block.attn,
            BottleNeck(
                encoder_block.attn.proj.out_features,
                activation,
                reduction,
            ),
        )

        encoder_block.mlp = torch.nn.Sequential(
            encoder_block.mlp,
            BottleNeck(
                encoder_block.mlp.lin2.out_features
                if hasattr(encoder_block.mlp, "lin2")
                else encoder_block.mlp.fc2.out_features,
                activation,
                reduction,
            ),
        )


def freeze_all(model):
    """
    Freeze the full model.
    """
    for _, param in model.named_parameters():
        param.requires_grad = False


def set_ntonly_trainable(model):
    """
    Train only the nuclei type decoder and tissue classifier head.
    """
    for name, param in model.named_parameters():
        if name.startswith("nuclei_type_maps_decoder") or name.startswith("classifier_head"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def set_lora_all_decoders_trainable(model):
    """
    Train LoRA adapters plus all decoder branches and classifier head.
    Freeze the original encoder parameters.
    """
    for name, param in model.named_parameters():
        if "encoder" in name and "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def is_encoder_adapter_parameter(name: str) -> bool:
    if not name.startswith("encoder"):
        return False
    return (
        "adapter" in name
        or "vera_" in name
        or "vera_adapter" in name
    )


def is_classifier_parameter(name: str) -> bool:
    return name.startswith("classifier_head")


def is_shared_decoder_parameter(name: str) -> bool:
    return name.startswith(SHARED_DECODER_PREFIXES)


def is_branch_decoder_parameter(name: str) -> bool:
    return name.startswith(BRANCH_DECODER_PREFIXES)


def is_decoder_parameter(name: str) -> bool:
    return is_shared_decoder_parameter(name) or is_branch_decoder_parameter(name)


def is_decoder_conv_adapter_parameter(name: str) -> bool:
    return "decoder_adapter" in name


def is_final_decoder_head_parameter(name: str) -> bool:
    return any(
        name.startswith(f"{prefix}.decoder0_header.2.")
        for prefix in BRANCH_DECODER_PREFIXES
    )


def is_last_stage_decoder_parameter(name: str) -> bool:
    if name.startswith("decoder0."):
        return True
    return any(
        name.startswith(f"{prefix}.decoder0_header.")
        for prefix in BRANCH_DECODER_PREFIXES
    )


def apply_peft_trainability(model, decoder_train_scope: str = "all") -> None:
    """
    Apply encoder-adapter plus decoder-scope trainability policy.

    The original encoder remains frozen. Encoder adapter params stay trainable.
    Decoder trainability is controlled by decoder_train_scope.
    """
    if decoder_train_scope is None:
        decoder_train_scope = "all"
    decoder_train_scope = str(decoder_train_scope).lower()
    if decoder_train_scope not in DECODER_TRAIN_SCOPES:
        raise ValueError(
            f"Unknown decoder_train_scope={decoder_train_scope!r}. "
            f"Allowed: {sorted(DECODER_TRAIN_SCOPES)}"
        )

    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        train = False

        if is_encoder_adapter_parameter(name):
            train = True
        elif decoder_train_scope == "all":
            train = (not name.startswith("encoder"))
        elif decoder_train_scope == "none":
            train = False
        elif decoder_train_scope == "nt_only":
            train = (
                name.startswith("nuclei_type_maps_decoder")
                or is_classifier_parameter(name)
            )
        elif decoder_train_scope == "heads_only":
            train = is_final_decoder_head_parameter(name) or is_classifier_parameter(name)
        elif decoder_train_scope == "last_stage":
            train = is_last_stage_decoder_parameter(name) or is_classifier_parameter(name)
        elif decoder_train_scope == "nt_header1_np_hv_heads":
            train = (
                name.startswith("nuclei_type_maps_decoder.decoder0_header.1.")
                or name.startswith("nuclei_type_maps_decoder.decoder0_header.2.")
                or name.startswith("nuclei_binary_map_decoder.decoder0_header.2.")
                or name.startswith("hv_map_decoder.decoder0_header.2.")
                or is_classifier_parameter(name)
            )
        elif decoder_train_scope == "nt_header_np_hv_heads":
            train = (
                name.startswith("nuclei_type_maps_decoder.decoder0_header.")
                or name.startswith("nuclei_binary_map_decoder.decoder0_header.2.")
                or name.startswith("hv_map_decoder.decoder0_header.2.")
                or is_classifier_parameter(name)
            )
        elif decoder_train_scope == "np_hv_heads_nt_all":
            train = (
                name.startswith("nuclei_type_maps_decoder")
                or name.startswith("nuclei_binary_map_decoder.decoder0_header.2.")
                or name.startswith("hv_map_decoder.decoder0_header.2.")
                or is_classifier_parameter(name)
            )
        elif decoder_train_scope == "conv_adapters":
            train = (
                is_decoder_conv_adapter_parameter(name)
                or is_final_decoder_head_parameter(name)
                or is_classifier_parameter(name)
            )

        param.requires_grad = bool(train)


def get_trainable_parameter_groups(model) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_encoder_adapter_parameter(name):
            groups["encoder adapters"].append(name)
        elif is_decoder_conv_adapter_parameter(name):
            groups["decoder adapters"].append(name)
        elif is_final_decoder_head_parameter(name):
            groups["decoder heads"].append(name)
        elif is_decoder_parameter(name):
            groups["decoder original weights"].append(name)
        elif is_classifier_parameter(name):
            groups["classifier head"].append(name)
        else:
            groups["other"].append(name)
    return dict(groups)


def format_trainable_parameter_groups(model, max_names_per_group=40) -> str:
    lines = ["Trainable module names by group:"]
    groups = get_trainable_parameter_groups(model)
    for group_name in [
        "encoder adapters",
        "decoder original weights",
        "decoder adapters",
        "decoder heads",
        "classifier head",
        "other",
    ]:
        names = groups.get(group_name, [])
        lines.append(f"[{group_name}] {len(names)} tensors")
        for name in names[:max_names_per_group]:
            lines.append(f"  {name}")
        if len(names) > max_names_per_group:
            lines.append(f"  ... ({len(names) - max_names_per_group} more)")
    return "\n".join(lines)


def set_lora_ntonly_trainable(model):
    """
    Train LoRA adapters plus nuclei type decoder and classifier head.
    Freeze original encoder, NP decoder, and HV decoder.
    """
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        elif name.startswith("nuclei_type_maps_decoder"):
            param.requires_grad = True
        elif name.startswith("classifier_head"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_model(model, adapter_type=None):
    """
    Backward-compatible alias.
    """
    freeze_all(model)
