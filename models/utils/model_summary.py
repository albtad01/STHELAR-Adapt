from collections import defaultdict


def _component_name(name: str) -> str:
    if name.startswith("encoder"):
        if "vera_" in name or "vera_adapter" in name:
            return "encoder_vera"
        if "attn.qkv.adapter_" in name:
            return "encoder_lora"
        if ".mlp.adapter_" in name:
            return "encoder_adaptformer"
        if "adapter" in name:
            return "encoder_other_adapter"
        return "encoder_base"
    if name.startswith(("decoder0", "decoder1", "decoder2", "decoder3")):
        if "decoder_adapter" in name:
            return "decoder_conv_adapter"
        return "shared_decoder"
    if name.startswith("nuclei_binary_map_decoder"):
        if "decoder_adapter" in name:
            return "decoder_conv_adapter"
        return "np_decoder"
    if name.startswith("hv_map_decoder"):
        if "decoder_adapter" in name:
            return "decoder_conv_adapter"
        return "hv_decoder"
    if name.startswith("nuclei_type_maps_decoder"):
        if "decoder_adapter" in name:
            return "decoder_conv_adapter"
        return "nt_decoder"
    if name.startswith("classifier_head"):
        return "classifier_head"
    return "other"


def log_trainable_parameter_report(
    model,
    logger,
    experiment_name="unknown",
    adapter_type="unknown",
    top_k=30,
):
    """
    Log total/trainable parameters and the largest trainable parameter groups.

    Parameters
    ----------
    model:
        PyTorch model.
    logger:
        Logger object with .info().
    experiment_name:
        Name of the experiment/run.
    adapter_type:
        Adapter/training mode.
    top_k:
        Number of largest trainable modules to print.
    """

    total_params = 0
    trainable_params = 0

    module_trainable = defaultdict(int)
    module_total = defaultdict(int)
    component_trainable = defaultdict(int)
    component_total = defaultdict(int)

    trainable_names = []

    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params

        # Group by first two name components, e.g.
        # encoder.blocks, nuclei_type_maps_decoder.decoder2, classifier_head
        parts = name.split(".")
        if len(parts) >= 2:
            group_name = ".".join(parts[:2])
        else:
            group_name = parts[0]

        module_total[group_name] += n_params
        component_total[_component_name(name)] += n_params

        if param.requires_grad:
            trainable_params += n_params
            module_trainable[group_name] += n_params
            component_trainable[_component_name(name)] += n_params
            trainable_names.append(name)

    trainable_ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINABLE PARAMETER REPORT")
    logger.info("=" * 80)
    logger.info(f"experiment_name: {experiment_name}")
    logger.info(f"adapter_type: {adapter_type}")
    logger.info(f"total_params: {total_params:,}")
    logger.info(f"trainable_params: {trainable_params:,}")
    logger.info(f"trainable_ratio_percent: {trainable_ratio:.4f}")
    logger.info(
        "Trainable parameter summary: "
        f"total={total_params:,}, "
        f"trainable={trainable_params:,}, "
        f"ratio={trainable_ratio:.4f}%"
    )
    logger.info("-" * 80)
    logger.info("Trainable parameter components:")
    component_order = [
        "encoder_base",
        "encoder_lora",
        "encoder_adaptformer",
        "encoder_vera",
        "encoder_other_adapter",
        "shared_decoder",
        "np_decoder",
        "hv_decoder",
        "nt_decoder",
        "decoder_conv_adapter",
        "classifier_head",
        "other",
    ]
    for component in component_order:
        n_total_component = component_total.get(component, 0)
        n_trainable_component = component_trainable.get(component, 0)
        if n_total_component == 0 and n_trainable_component == 0:
            continue
        ratio_component = (
            100.0 * n_trainable_component / n_total_component
            if n_total_component > 0
            else 0.0
        )
        logger.info(
            f"{component:28s} "
            f"trainable={n_trainable_component:,} "
            f"total={n_total_component:,} "
            f"ratio={ratio_component:.2f}%"
        )

    logger.info("-" * 80)

    logger.info(f"Top {top_k} trainable parameter groups:")
    sorted_groups = sorted(
        module_trainable.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for group_name, n_trainable in sorted_groups[:top_k]:
        n_total_group = module_total[group_name]
        ratio_group = 100.0 * n_trainable / n_total_group if n_total_group > 0 else 0.0
        logger.info(
            f"{group_name:60s} "
            f"trainable={n_trainable:,} "
            f"total={n_total_group:,} "
            f"ratio={ratio_group:.2f}%"
        )

    logger.info("-" * 80)
    logger.info("First trainable parameter names:")
    for name in trainable_names[:top_k]:
        logger.info(f"  {name}")

    if len(trainable_names) > top_k:
        logger.info(f"  ... ({len(trainable_names) - top_k} more)")

    logger.info("=" * 80)
    logger.info("")

    return {
        "experiment_name": experiment_name,
        "adapter_type": adapter_type,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio_percent": trainable_ratio,
        "trainable_groups": dict(module_trainable),
        "trainable_components": dict(component_trainable),
        "trainable_names": trainable_names,
    }
