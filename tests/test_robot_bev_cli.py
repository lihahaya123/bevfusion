import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


SCENES = (
    "apartment_0",
    "apartment_1",
    "apartment_2",
    "frl_apartment_0",
    "frl_apartment_1",
    "frl_apartment_2",
    "frl_apartment_3",
    "frl_apartment_4",
    "frl_apartment_5",
    "hotel_0",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "room_0",
    "room_1",
    "room_2",
)
SPLITS = {
    "train": [
        "apartment_0",
        "apartment_1",
        "frl_apartment_0",
        "frl_apartment_1",
        "frl_apartment_2",
        "frl_apartment_3",
        "hotel_0",
        "office_0",
        "office_2",
        "office_3",
        "room_0",
        "room_2",
        "room_1",
        "frl_apartment_4",
    ],
    "val": ["apartment_2", "office_1"],
    "test": ["frl_apartment_5", "office_4"],
}
SMOKE_SCENES = (
    "hotel_0",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "room_0",
    "room_1",
    "room_2",
)


def _build_dataset(root: Path) -> None:
    writer = RobotBEVWriter(
        root=root,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"fixture": True},
    )
    labels = np.zeros((6, 150, 150), dtype=np.uint8)
    labels[0, 45:55, 70:80] = 1
    writer.write_frame(
        "scene_a",
        "train",
        FramePayload(
            frame_id=0,
            timestamp=1_000_000,
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            points=np.array([[0.0, 0.0, 2.0, 0.5, 0.0]], dtype=np.float32),
            bev_labels=labels,
            observed_mask=np.ones((150, 150), dtype=np.uint8),
            class_validity=np.ones((6,), dtype=np.uint8),
            cam_intrinsic=np.array(
                [[10, 0, 6], [0, 10, 4], [0, 0, 1]], dtype=np.float32
            ),
            camera2base=np.eye(4, dtype=np.float32),
            lidar2base=np.eye(4, dtype=np.float32),
            map_from_base=np.eye(4, dtype=np.float32),
        ),
    )
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()


def _run_cli(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "data_generation.robot_bev.cli.validate_dataset",
            *(str(arg) for arg in args),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validation_cli_help():
    completed = _run_cli("--help")

    assert completed.returncode == 0
    assert "--root" in completed.stdout
    assert "--split" in completed.stdout
    assert "--geometry-scene" in completed.stdout
    assert "--geometry-frame" in completed.stdout


def test_validation_cli_prints_json_report_and_geometry_paths(tmp_path):
    _build_dataset(tmp_path)

    completed = _run_cli(
        "--root",
        tmp_path,
        "--split",
        "train",
        "--geometry-scene",
        "scene_a",
        "--geometry-frame",
        0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["frame_counts"] == {"train": 1, "val": 0, "test": 0}
    assert isinstance(report["warnings"], list)
    assert [Path(path).name for path in report["geometry_diagnostics"]] == [
        "000000_rgb_point_overlay.png",
        "000000_bev_overlay.png",
        "000000_aligned_sweeps.png",
        "000000_overview.png",
    ]
    assert all(Path(path).is_file() for path in report["geometry_diagnostics"])


def test_validation_cli_prints_json_error_to_stderr(tmp_path):
    missing_root = tmp_path / "missing"

    completed = _run_cli("--root", missing_root)

    assert completed.returncode == 1
    assert completed.stdout == ""
    error = json.loads(completed.stderr)
    assert error["valid"] is False
    assert error["error_type"] == "DatasetValidationError"
    assert str(missing_root) in error["error"]


def test_validation_cli_prints_json_error_for_invalid_arguments(tmp_path):
    completed = _run_cli("--root", tmp_path, "--split", "invalid")

    assert completed.returncode != 0
    assert completed.stdout == ""
    error = json.loads(completed.stderr)
    assert error["valid"] is False
    assert error["error_type"] == "ArgumentParseError"
    assert "invalid choice" in error["error"]


def test_replica_scene_examples_are_exact_and_disjoint():
    config_root = Path("data_generation/robot_bev/configs")
    scene_lines = tuple(
        line.strip()
        for line in (config_root / "replica_scenes.txt").read_text().splitlines()
        if line.strip()
    )
    splits = json.loads(
        (config_root / "replica_splits.example.json").read_text()
    )

    assert scene_lines == SCENES
    assert splits == SPLITS
    assert {name: len(scenes) for name, scenes in splits.items()} == {
        "train": 14,
        "val": 2,
        "test": 2,
    }
    assigned = [scene for scenes in splits.values() for scene in scenes]
    assert len(assigned) == len(set(assigned)) == 18
    assert set(assigned) == set(scene_lines)


def _bash_blocks(markdown: str):
    return re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL)


def test_documented_commands_match_supported_smoke_and_production_contracts():
    docs = {
        path: path.read_text(encoding="utf-8")
        for path in (
            Path("data_generation/robot_bev/README.md"),
            Path("data_generation/robot_bev/docs/schema_v3.md"),
            Path("data_generation/robot_bev/docs/habitat_replica.md"),
            Path("data_generation/robot_bev/docs/add_new_source.md"),
            Path("data_generation/robot_bev/docs/quality_checks.md"),
        )
    }
    command_text = "\n".join(
        block for markdown in docs.values() for block in _bash_blocks(markdown)
    )

    assert "--save-visualization" not in command_text
    assert "--save-ply" not in command_text
    assert "/mnt/u/ubuntu/workspace/dataset" not in command_text

    generation_commands = [
        block
        for block in _bash_blocks(
            docs[Path("data_generation/robot_bev/docs/habitat_replica.md")]
        )
        if "cli.generate_replica" in block
    ]
    assert len(generation_commands) == 3
    quick, smoke, production = generation_commands
    for command in generation_commands:
        assert "--gpu-id 0" in command
        assert "--disable-physics" in command
        assert "--recompute-navmesh" in command
        assert "--split-file data_generation/robot_bev/configs/replica_splits.example.json" in command

    assert "--scene office_1" in quick
    assert "--num-frames 10" in quick
    assert "--dataset-id replica_robot_bev_v3_quick" in quick

    assert "--dataset-id replica_robot_bev_v3" in smoke
    assert "--num-frames 10" in smoke
    assert "--scenes " in smoke
    assert all(scene in smoke for scene in SMOKE_SCENES)

    assert "--dataset-id replica_robot_bev_v3" in production
    assert "--scenes-file data_generation/robot_bev/configs/replica_scenes.txt" in production
    assert "--num-frames 600" in production

    validation_commands = [
        block
        for block in _bash_blocks(
            docs[Path("data_generation/robot_bev/docs/habitat_replica.md")]
        )
        if "cli.validate_dataset" in block
    ]
    validation_text = "\n".join(validation_commands)
    for split in ("train", "val", "test"):
        assert f"--split {split}" in validation_text
    for scene in ("office_0", "office_1", "office_4"):
        assert f"--geometry-scene {scene}" in validation_text
