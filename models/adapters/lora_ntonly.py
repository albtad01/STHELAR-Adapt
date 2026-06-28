from typing import Iterable, Optional


def set_lora_ntonly_trainable(
    model,
    train_classifier_head: bool = True,
    extra_trainable_prefixes: Optional[Iterable[str]] = None,
) -> None:
    """
    Training mode for LoRA + NTOnly.

    It freezes the whole model, then enables training only for:
      - LoRA/adapter parameters inside the encoder;
      - nuclei_type_maps_decoder;
      - classifier_head, if present and requested.

    This is intended for STHELAR adaptation where:
      - NP/HV detection and instance separation are already reasonably strong;
      - the main bottleneck is nuclei type classification.
    """

    # 1. Freeze everything.
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_prefixes = ["nuclei_type_maps_decoder"]

    if train_classifier_head:
        trainable_prefixes.append("classifier_head")

    if extra_trainable_prefixes is not None:
        trainable_prefixes.extend(list(extra_trainable_prefixes))

    # 2. Unfreeze LoRA/adapters + NT decoder + classifier head.
    for name, param in model.named_parameters():
        is_adapter_param = "adapter" in name

        is_allowed_prefix = any(
            name.startswith(prefix) for prefix in trainable_prefixes
        )

        if is_adapter_param or is_allowed_prefix:
            param.requires_grad = True


def print_lora_ntonly_trainable_parameters(model, logger=None) -> None:
    """
    Small debug utility to check that the right parameters are trainable.
    """

    lines = []
    lines.append("\nTrainable parameters for lora_ntonly mode:")
    lines.append("-" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params

        if param.requires_grad:
            trainable_params += n_params
            lines.append(f"{name:90s} {n_params}")

    ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    lines.append("-" * 80)
    lines.append(f"Total params:      {total_params:,}")
    lines.append(f"Trainable params:  {trainable_params:,}")
    lines.append(f"Trainable ratio:   {ratio:.4f}%")
    lines.append("")

    text = "\n".join(lines)

    if logger is not None:
        logger.info(text)
    else:
        print(text)
