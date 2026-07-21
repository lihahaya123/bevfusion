import pickle

import numpy as np

from tools.data_converter.robot_bev_converter import convert_split


def test_converter_preserves_relative_paths_and_builds_sweeps(canonical_root):
    output = convert_split(canonical_root, "train", max_sweeps=5)
    with output.open("rb") as handle:
        payload = pickle.load(handle)
    first, second = payload["infos"][:2]
    assert first["lidar_path"] == "scene_a/points/000000.bin"
    assert first["cams"]["CAM_FRONT"]["data_path"] == (
        "scene_a/images/000000.png"
    )
    assert first["sweeps"] == []
    assert len(second["sweeps"]) == 1
    np.testing.assert_allclose(
        second["sweeps"][0]["sensor2lidar_translation"],
        np.array([-0.1, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    assert second["bev_observed_mask_path"].endswith(
        "bev_observed_masks/000001.npy"
    )
    assert payload["metadata"]["version"] == "robot-bev-v4"


def test_converter_is_byte_deterministic(canonical_root):
    first = convert_split(canonical_root, "train", max_sweeps=5).read_bytes()
    second = convert_split(canonical_root, "train", max_sweeps=5).read_bytes()
    assert first == second
