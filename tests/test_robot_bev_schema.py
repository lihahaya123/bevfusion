from pathlib import Path

import numpy as np
import pytest

from data_generation.robot_bev.schema import (
    BEV_SHAPE,
    MAP_CLASSES,
    SchemaError,
    canonical_token,
    effective_supervision_mask,
    normalize_relative_path,
)


def test_schema_constants_are_fixed():
    assert MAP_CLASSES == (
        "floor",
        "carpet",
        "obstacle",
        "wall",
        "furniture",
        "other",
    )
    assert BEV_SHAPE == (6, 150, 150)


def test_relative_path_is_portable_and_cannot_escape_root():
    assert normalize_relative_path(Path("office_0/images/000012.png")) == (
        "office_0/images/000012.png"
    )
    assert normalize_relative_path("office_0\\images\\000012.png") == (
        "office_0/images/000012.png"
    )
    for invalid in ("/tmp/frame.png", "../frame.png", "C:/frame.png", ""):
        with pytest.raises(SchemaError):
            normalize_relative_path(invalid)


def test_effective_mask_broadcasts_without_duplicate_storage():
    observed = np.zeros((150, 150), dtype=np.uint8)
    observed[10:20, 30:40] = 1
    class_validity = np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8)
    effective = effective_supervision_mask(observed, class_validity)
    assert effective.shape == (6, 150, 150)
    assert effective.dtype == np.uint8
    assert effective[0].sum() == 100
    assert effective[1].sum() == 0


def test_optional_per_class_mask_is_intersected():
    observed = np.ones((150, 150), dtype=np.uint8)
    class_validity = np.ones((6,), dtype=np.uint8)
    regional = np.ones((6, 150, 150), dtype=np.uint8)
    regional[4, :, 75:] = 0
    effective = effective_supervision_mask(observed, class_validity, regional)
    assert effective[4, :, 75:].sum() == 0
    assert effective[4, :, :75].sum() == 150 * 75


def test_token_is_dataset_scene_and_frame_scoped():
    assert canonical_token("replica_v3", "office_0", 12) == (
        "replica_v3:office_0:000012"
    )
