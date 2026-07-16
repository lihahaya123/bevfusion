from typing import Dict, Iterable, List, Optional

import torch

__all__ = ["load_checkpoint_selectively"]


def _log(logger, message: str) -> None:
    if logger is None:
        print(message)
    else:
        logger.info(message)


def load_checkpoint_selectively(
    model: torch.nn.Module,
    filename: str,
    *,
    skip_prefixes: Optional[Iterable[str]] = None,
    map_location: str = "cpu",
    logger=None,
) -> Dict[str, List]:
    """Load matching checkpoint tensors and skip unsafe mismatches.

    This is useful when fine-tuning from a source-domain checkpoint whose
    architecture mostly matches the current model, but some parameters have
    different shapes or same-shape-but-different-semantics heads.
    """

    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    skip_prefixes = tuple(skip_prefixes or ())

    matched = {}
    skipped_prefix = []
    skipped_shape = []
    unexpected = []

    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_prefix.append(key)
            continue
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue
        matched[key] = value

    incompatible = model.load_state_dict(matched, strict=False)
    report = {
        "loaded": sorted(matched),
        "skipped_prefix": skipped_prefix,
        "skipped_shape": skipped_shape,
        "unexpected": unexpected,
        "missing": incompatible.missing_keys,
    }

    _log(
        logger,
        "Selective checkpoint load: "
        f"loaded={len(report['loaded'])}, "
        f"skipped_prefix={len(skipped_prefix)}, "
        f"skipped_shape={len(skipped_shape)}, "
        f"unexpected={len(unexpected)}, "
        f"missing={len(incompatible.missing_keys)}",
    )
    if skipped_prefix:
        _log(logger, f"Skipped by prefix: {skipped_prefix[:20]}")
    if skipped_shape:
        _log(logger, f"Skipped by shape: {skipped_shape[:20]}")
    if unexpected:
        _log(logger, f"Unexpected checkpoint keys: {unexpected[:20]}")

    return report
