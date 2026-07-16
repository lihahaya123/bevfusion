from pathlib import PurePosixPath
from typing import Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_NAME = "robot_bev_dataset"
SCHEMA_VERSION = 3
MAP_CLASSES: Tuple[str, ...] = (
    "floor",
    "carpet",
    "obstacle",
    "wall",
    "furniture",
    "other",
)
BEV_XBOUND = (0.0, 3.0, 0.02)
BEV_YBOUND = (-1.5, 1.5, 0.02)
BEV_SHAPE = (6, 150, 150)
OBSERVED_MASK_SHAPE = (150, 150)
POINT_DIMENSIONS = ("x", "y", "z", "intensity", "time")


class SchemaError(ValueError):
    pass


def canonical_token(dataset_id: str, scene_id: str, frame_id: int) -> str:
    if not dataset_id or not scene_id or frame_id < 0:
        raise SchemaError("dataset_id and scene_id are required; frame_id must be non-negative")
    if ":" in dataset_id or ":" in scene_id:
        raise SchemaError("dataset_id and scene_id must not contain ':'")
    return f"{dataset_id}:{scene_id}:{frame_id:06d}"


def normalize_relative_path(path: Union[str, PurePosixPath]) -> str:
    text = str(path).replace("\\", "/")
    candidate = PurePosixPath(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise SchemaError(f"path must remain inside the dataset root: {path!r}")
    if candidate.parts and candidate.parts[0].endswith(":"):
        raise SchemaError(f"Windows absolute paths are forbidden: {path!r}")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise SchemaError(f"empty relative path is forbidden: {path!r}")
    return normalized


def _binary_uint8(value: np.ndarray, shape: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != tuple(shape):
        raise SchemaError(f"{name} shape {array.shape} != {tuple(shape)}")
    if array.dtype != np.uint8:
        raise SchemaError(f"{name} dtype {array.dtype} != uint8")
    if not np.isin(array, (0, 1)).all():
        raise SchemaError(f"{name} must be binary")
    return array


def effective_supervision_mask(
    observed_mask: np.ndarray,
    class_validity: np.ndarray,
    per_class_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    observed = _binary_uint8(observed_mask, OBSERVED_MASK_SHAPE, "observed_mask")
    validity = _binary_uint8(class_validity, (len(MAP_CLASSES),), "class_validity")
    effective = observed[None, :, :] * validity[:, None, None]
    if per_class_mask is not None:
        regional = _binary_uint8(per_class_mask, BEV_SHAPE, "per_class_mask")
        effective = effective * regional
    return effective.astype(np.uint8, copy=False)
