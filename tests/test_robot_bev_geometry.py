import numpy as np
from PIL import Image

from data_generation.robot_bev.geometry_checks import (
    history_to_current_lidar,
    points_to_bev_cells,
    project_lidar_to_image,
    write_geometry_diagnostics,
)
from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


def test_opencv_projection_uses_z_forward():
    points = np.array(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32
    )
    intrinsic = np.array(
        [[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=np.float32
    )
    uv, valid = project_lidar_to_image(
        points, np.eye(4, dtype=np.float32), intrinsic, (80, 100)
    )
    np.testing.assert_allclose(uv[0], [50, 40])
    assert valid.tolist() == [True, False]


def test_base_forward_left_maps_to_row_column():
    points = np.array(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32
    )
    rows, cols, valid = points_to_bev_cells(points)
    assert (rows[0], cols[0], valid[0]) == (50, 75, True)
    assert (rows[1], cols[1], valid[1]) == (100, 125, True)


def test_history_transform_matches_contract_formula():
    current_pose = np.eye(4, dtype=np.float32)
    current_pose[0, 3] = 1.0
    history_pose = np.eye(4, dtype=np.float32)
    transform = history_to_current_lidar(
        current_pose,
        np.eye(4, dtype=np.float32),
        history_pose,
        np.eye(4, dtype=np.float32),
    )
    np.testing.assert_allclose(transform[:3, 3], [-1.0, 0.0, 0.0])


def _payload(frame_id: int) -> FramePayload:
    labels = np.zeros((6, 150, 150), dtype=np.uint8)
    labels[0, 45:55, 70:80] = 1
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = frame_id * 0.1
    return FramePayload(
        frame_id=frame_id,
        timestamp=1_000_000 + frame_id * 100_000,
        rgb=np.zeros((80, 100, 3), dtype=np.uint8),
        points=np.array(
            [[0.0, 0.0, 2.0, 0.5, 0.0], [1.0, 0.0, 0.1, 1.0, 0.0]],
            dtype=np.float32,
        ),
        bev_labels=labels,
        observed_mask=np.ones((150, 150), dtype=np.uint8),
        class_validity=np.ones((6,), dtype=np.uint8),
        cam_intrinsic=np.array(
            [[50.0, 0.0, 50.0], [0.0, 50.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        camera2base=np.eye(4, dtype=np.float32),
        lidar2base=np.eye(4, dtype=np.float32),
        map_from_base=pose,
    )


def test_write_geometry_diagnostics_writes_rgb_pngs(tmp_path):
    writer = RobotBEVWriter(
        root=tmp_path,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"width": 100, "height": 80},
    )
    writer.write_frame("scene_a", "train", _payload(0))
    writer.write_frame("scene_a", "train", _payload(1))
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()

    paths = write_geometry_diagnostics(
        tmp_path, "scene_a", frame_id=1, history_count=1
    )

    diagnostics = tmp_path / "diagnostics" / "scene_a"
    assert {path.name for path in paths} == {
        "000001_aligned_sweeps.png",
        "000001_bev_overlay.png",
        "000001_overview.png",
        "000001_rgb_point_overlay.png",
    }
    assert set(diagnostics.iterdir()) == set(paths)
    for path in paths:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
