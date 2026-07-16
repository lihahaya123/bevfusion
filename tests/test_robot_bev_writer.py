import json
import pickle
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from data_generation.robot_bev.schema import SchemaError
from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


def make_payload(frame_id: int) -> FramePayload:
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


def make_writer(root) -> RobotBEVWriter:
    return RobotBEVWriter(
        root=root,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"width": 12, "height": 8},
    )


def test_writer_creates_root_relative_canonical_indexes(tmp_path):
    writer = RobotBEVWriter(
        root=tmp_path,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"width": 12, "height": 8},
    )
    writer.write_frame("scene_a", "train", make_payload(0))
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()

    with (tmp_path / "robot_infos_train.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    info = payload["infos"][0]
    assert info["image_path"] == "scene_a/images/000000.png"
    assert info["token"] == "fixture_v3:scene_a:000000"
    assert info["class_validity"].tolist() == [1, 1, 1, 1, 1, 1]
    assert "sweeps" not in info
    metadata = json.loads((tmp_path / "dataset_metadata.json").read_text())
    assert metadata["schema_version"] == 3


def test_writer_refuses_resume_when_generation_contract_changes(tmp_path):
    common = dict(
        root=tmp_path,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
    )
    RobotBEVWriter(generation_parameters={"hfov": 120.0}, **common)
    try:
        RobotBEVWriter(
            generation_parameters={"hfov": 90.0}, resume=True, **common
        )
    except RuntimeError as error:
        assert "generation fingerprint mismatch" in str(error)
    else:
        raise AssertionError("resume must reject changed generation parameters")


def test_optional_uint16_pngs_round_trip_boundary_values(tmp_path):
    depth_mm = np.zeros((8, 12), dtype=np.uint16)
    depth_mm[0, 1] = 65535
    semantics = np.full((8, 12), 65535, dtype=np.uint16)
    semantics[0, 1] = 0
    frame = replace(
        make_payload(0), depth_mm=depth_mm, semantics=semantics
    )

    writer = make_writer(tmp_path)
    record = writer.write_frame("scene_a", "train", frame)

    for key, expected in (
        ("depth_path", depth_mm),
        ("semantic_path", semantics),
    ):
        path = tmp_path / record[key]
        encoded = path.read_bytes()
        assert encoded[24] == 16
        assert encoded[25] == 0
        with Image.open(path) as image:
            actual = np.asarray(image)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("depth_mm", np.full((8, 12), 65536, dtype=np.int32)),
        ("semantics", np.zeros((8, 12), dtype=np.uint8)),
    ),
)
def test_optional_pngs_reject_unsupported_dtype_or_range(
    tmp_path, field, value
):
    writer = make_writer(tmp_path)
    frame = replace(make_payload(0), **{field: value})

    with pytest.raises(SchemaError, match=rf"{field}.*uint16"):
        writer.write_frame("scene_a", "train", frame)

    assert not (tmp_path / "scene_a" / "manifest.jsonl").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "points",
            np.full((1, 5), np.finfo(np.float64).max, dtype=np.float64),
        ),
        (
            "camera2base",
            np.full((4, 4), np.finfo(np.float64).max, dtype=np.float64),
        ),
        ("cam_intrinsic", np.eye(3, dtype=np.complex64) * (1 + 1j)),
    ),
)
def test_writer_rejects_values_not_representable_as_finite_float32(
    tmp_path, field, value
):
    writer = make_writer(tmp_path)
    frame = replace(make_payload(0), **{field: value})

    with pytest.raises(SchemaError, match=rf"{field}.*float32"):
        writer.write_frame("scene_a", "train", frame)

    assert not (tmp_path / "scene_a" / "manifest.jsonl").exists()
