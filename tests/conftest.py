from dataclasses import replace

import numpy as np
import pytest

from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


def _frame(frame_id: int) -> FramePayload:
    return FramePayload(
        frame_id=frame_id,
        timestamp=1_000_000 + frame_id * 100_000,
        rgb=np.zeros((8, 12, 3), dtype=np.uint8),
        points=np.array([[1.0, 0.0, 0.1, 0.0, 0.0]], dtype=np.float32),
        bev_labels=np.zeros((6, 150, 150), dtype=np.uint8),
        observed_mask=np.ones((150, 150), dtype=np.uint8),
        class_validity=np.ones((6,), dtype=np.uint8),
        cam_intrinsic=np.eye(3, dtype=np.float32),
        camera2base=np.eye(4, dtype=np.float32),
        lidar2base=np.eye(4, dtype=np.float32),
        map_from_base=np.eye(4, dtype=np.float32),
    )


@pytest.fixture
def canonical_root(tmp_path):
    writer = RobotBEVWriter(
        root=tmp_path,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"fixture": True},
    )
    writer.write_frame("scene_a", "train", _frame(0))
    moved = replace(_frame(1), map_from_base=np.eye(4, dtype=np.float32))
    moved.map_from_base[0, 3] = 0.1
    writer.write_frame("scene_a", "train", moved)
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()
    return tmp_path
