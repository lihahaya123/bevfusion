import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_generation.robot_bev.sources.selfcollect import (
    DEFAULT_IGNORED_SEMANTIC_IDS,
    DEFAULT_SEMANTIC_ID_TO_CLASS,
    SelfCollectFrame,
    estimate_camera2base_from_floor,
    load_intrinsic,
    load_semantic_policy,
    make_parser,
    run_generation,
    split_frames,
)
from data_generation.robot_bev.validator import validate_dataset


def test_conservative_default_semantic_policy():
    policy = load_semantic_policy()

    assert policy.id_to_class == DEFAULT_SEMANTIC_ID_TO_CLASS
    assert policy.ignore_ids == DEFAULT_IGNORED_SEMANTIC_IDS
    assert policy.id_to_class == {
        0: "carpet",
        1: "furniture",
        2: "door",
        4: "floor",
        6: "clutter",
        10: "furniture",
        12: "clutter",
        15: "wall",
        18: "furniture",
    }
    assert {3, 5, 7, 8, 9, 11, 13, 14, 16, 17, 19, 255} <= set(
        policy.ignore_ids
    )


def test_intrinsic_parser_accepts_cpp_float_suffix(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("347.6f\n347.7f\n316.0f\n240.8f", encoding="utf-8")

    intrinsic = load_intrinsic(path)

    np.testing.assert_allclose(
        intrinsic,
        [[347.6, 0.0, 316.0], [0.0, 347.7, 240.8], [0.0, 0.0, 1.0]],
    )


def test_floor_plane_estimates_camera_height(tmp_path):
    height, width = 40, 40
    fy = 20.0
    cy = 10.0
    rows = np.arange(height, dtype=np.float32)[:, None]
    depth_m = np.zeros((height, width), dtype=np.float32)
    below_horizon = rows[:, 0] > cy
    depth_m[below_horizon, :] = fy / (rows[below_horizon] - cy)
    depth_mm = np.rint(depth_m * 1000.0).astype(np.uint16)
    labels = np.full((height, width), 255, dtype=np.uint8)
    labels[below_horizon, :] = 4
    rgb_path = tmp_path / "1_rgb.jpg"
    depth_path = tmp_path / "1_rgb.png"
    label_path = tmp_path / "1_label.png"
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(depth_mm).save(depth_path)
    Image.fromarray(labels).save(label_path)
    frame = SelfCollectFrame(1, rgb_path, depth_path, label_path)
    intrinsic = np.array(
        [[20.0, 0.0, 19.5], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    camera2base, diagnostics = estimate_camera2base_from_floor(
        [frame], intrinsic, boundary_radius=0, sample_stride=1
    )

    assert abs(float(camera2base[2, 3]) - 1.0) < 0.01
    np.testing.assert_allclose(camera2base[2, :3], [0.0, -1.0, 0.0], atol=1e-3)
    assert diagnostics["inlier_ratio"] > 0.95


def _write_tiny_source(root: Path, frame_count: int = 1) -> Path:
    for name in ("Left", "Depth", "Label"):
        (root / name).mkdir(parents=True)
    root.joinpath("in.txt").write_text("10f\n10f\n3.5f\n3.5f", encoding="utf-8")
    trajectory_rows = []
    for index in range(frame_count):
        raw_frame_id = index + 1
        trajectory_rows.append(
            f"{raw_frame_id * 1000000} {index * 0.1} 0 0 0 0 0 1"
        )
        stem = f"{raw_frame_id}_rgb_left_r_s"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            root / "Left" / f"{stem}.png"
        )
        Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16)).save(
            root / "Depth" / f"{stem}.png"
        )
        labels = np.full((8, 8), 4, dtype=np.uint8)
        labels[4, 4] = 8  # PERSON is deliberately unsupported.
        Image.fromarray(labels).save(root / "Label" / f"{stem}.png")
    root.joinpath("CameraTrajectory.txt").write_text(
        "\n".join(trajectory_rows) + "\n", encoding="utf-8"
    )
    camera2base = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    path = root / "camera2base.txt"
    np.savetxt(path, camera2base)
    return path


def test_selfcollect_generation_writes_valid_canonical_dataset(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    camera2base = _write_tiny_source(source)
    args = make_parser().parse_args(
        [
            "--dataset",
            str(source),
            "--dataset-id",
            "selfcollect_fixture",
            "--scene",
            "fixture_scene",
            "--output-dir",
            str(output),
            "--camera2base",
            str(camera2base),
            "--depth-stride",
            "1",
            "--max-points",
            "100",
            "--min-points",
            "1",
            "--min-semantic-coverage",
            "0",
            "--min-observed-coverage",
            "0",
            "--label-boundary-pixels",
            "0",
            "--ignore-dilation-cells",
            "0",
        ]
    )

    run_generation(args)
    report = validate_dataset(output)

    assert report.valid is True
    assert report.frame_counts == {"train": 1, "val": 0, "test": 0}
    assert (output / "fixture_scene" / "bev_masks" / "000000.npy").is_file()
    observed_path = (
        output / "fixture_scene" / "bev_observed_masks" / "000000.npy"
    )
    assert observed_path.is_file()
    observed = np.load(observed_path)
    # Pixel (u=4,v=4), Z=1 m projects to base (x=1,y=-0.05,z=0.65).
    # Its BEV cell must be invalid even if neighbouring FLOOR rays observe it.
    assert observed[50, 72] == 0


def test_sampled_split_uses_periodic_seven_one_one_allocation():
    frames = [
        SelfCollectFrame(
            index,
            Path(f"rgb_{index}"),
            Path(f"depth_{index}"),
            Path(f"label_{index}"),
        )
        for index in range(690)
    ]

    splits = split_frames(frames, split_mode="sampled", split_ratios=(7, 1, 1))

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 538,
        "val": 76,
        "test": 76,
    }
    assert [frame.raw_frame_id for frame in splits["train"][:7]] == list(range(7))
    assert splits["val"][0].raw_frame_id == 7
    assert splits["test"][0].raw_frame_id == 8
    assigned = [frame.raw_frame_id for items in splits.values() for frame in items]
    assert sorted(assigned) == list(range(690))


def test_sampled_generation_writes_three_split_scenes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    camera2base = _write_tiny_source(source, frame_count=9)
    args = make_parser().parse_args(
        [
            "--dataset",
            str(source),
            "--dataset-id",
            "selfcollect_sampled_fixture",
            "--scene",
            "fixture_scene",
            "--split-mode",
            "sampled",
            "--split-ratios",
            "7",
            "1",
            "1",
            "--output-dir",
            str(output),
            "--camera2base",
            str(camera2base),
            "--depth-stride",
            "1",
            "--max-points",
            "100",
            "--min-points",
            "1",
            "--min-semantic-coverage",
            "0",
            "--min-observed-coverage",
            "0",
            "--label-boundary-pixels",
            "0",
            "--ignore-dilation-cells",
            "0",
        ]
    )

    run_generation(args)
    report = validate_dataset(output)

    assert report.valid is True
    assert report.frame_counts == {"train": 7, "val": 1, "test": 1}
    metadata = json.loads((output / "dataset_metadata.json").read_text())
    assert metadata["generation_parameters"]["split_mode"] == "sampled"
    split_index = json.loads((output / "splits.json").read_text())
    expected = {
        "train": ("fixture_scene_train", [1, 2, 3, 4, 5, 6, 7]),
        "val": ("fixture_scene_val", [8]),
        "test": ("fixture_scene_test", [9]),
    }
    for split, (scene, raw_ids) in expected.items():
        manifest = [
            json.loads(line)
            for line in (output / scene / "manifest.jsonl").read_text().splitlines()
        ]
        assert [row["frame_id"] for row in manifest] == list(range(len(raw_ids)))
        assert [row["raw_frame_id"] for row in manifest] == raw_ids
        assert split_index[split] == [scene]
