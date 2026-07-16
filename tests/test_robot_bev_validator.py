import json
import pickle

import numpy as np
import pytest

from data_generation.robot_bev.validator import (
    DatasetValidationError,
    validate_dataset,
)
from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


def build_dataset(root, frame_count=1):
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
    for frame_id in range(frame_count):
        frame = FramePayload(
            frame_id=frame_id,
            timestamp=1_000_000 + frame_id * 100_000,
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            points=np.zeros((1, 5), dtype=np.float32),
            bev_labels=np.zeros((6, 150, 150), dtype=np.uint8),
            observed_mask=np.ones((150, 150), dtype=np.uint8),
            class_validity=np.ones((6,), dtype=np.uint8),
            cam_intrinsic=np.array(
                [[10, 0, 6], [0, 10, 4], [0, 0, 1]], dtype=np.float32
            ),
            camera2base=np.eye(4, dtype=np.float32),
            lidar2base=np.eye(4, dtype=np.float32),
            map_from_base=np.eye(4, dtype=np.float32),
        )
        writer.write_frame("scene_a", "train", frame)
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()


def test_validator_accepts_a_complete_dataset(tmp_path):
    build_dataset(tmp_path)
    report = validate_dataset(tmp_path)
    assert report.valid
    assert report.frame_counts == {"train": 1, "val": 0, "test": 0}


def test_validator_reports_context_for_path_escape(tmp_path):
    build_dataset(tmp_path)
    manifest_path = tmp_path / "scene_a" / "manifest.jsonl"
    record = json.loads(manifest_path.read_text().strip())
    record["image_path"] = "../escape.png"
    manifest_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(DatasetValidationError) as caught:
        validate_dataset(tmp_path)
    message = str(caught.value)
    assert "fixture_v3" in message
    assert "scene_a" in message
    assert "image_path" in message


def _rewrite_pickle(path, update):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    update(payload)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _rewrite_all_indexes(root, update):
    for path in (
        root / "robot_infos_train.pkl",
        root / "scene_a" / "scene_infos.pkl",
        root / "scene_a" / "robot_infos_train.pkl",
    ):
        _rewrite_pickle(path, update)


def _rewrite_manifest(root, update):
    path = root / "scene_a" / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    update(records)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _rewrite_json(path, update):
    payload = json.loads(path.read_text())
    update(payload)
    path.write_text(json.dumps(payload) + "\n")


def _corrupt_mask_dtype(root):
    np.save(
        root / "scene_a" / "bev_observed_masks" / "000000.npy",
        np.ones((150, 150), dtype=np.int16),
        allow_pickle=False,
    )


def _corrupt_mask_shape(root):
    np.save(
        root / "scene_a" / "bev_observed_masks" / "000000.npy",
        np.ones((149, 150), dtype=np.uint8),
        allow_pickle=False,
    )


def _corrupt_rotation_determinant(root):
    reflection = np.eye(4, dtype=np.float32)
    reflection[0, 0] = -1.0

    def update(payload):
        payload["infos"][0]["camera2base"] = reflection.copy()

    _rewrite_all_indexes(root, update)
    manifest_path = root / "scene_a" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    records[0]["camera2base"] = reflection.tolist()
    manifest_path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _corrupt_point_byte_length(root):
    path = root / "scene_a" / "points" / "000000.bin"
    path.write_bytes(path.read_bytes() + b"\x00")


def _corrupt_token_uniqueness(root):
    def update(payload):
        payload["infos"][1]["token"] = payload["infos"][0]["token"]

    _rewrite_pickle(root / "robot_infos_train.pkl", update)


def _corrupt_split_overlap(root):
    path = root / "splits.json"
    splits = json.loads(path.read_text())
    splits["val"].append("scene_a")
    path.write_text(json.dumps(splits) + "\n")


@pytest.mark.parametrize(
    ("corrupt", "field_name", "frame_count"),
    (
        (_corrupt_mask_dtype, "bev_observed_mask_path", 1),
        (_corrupt_mask_shape, "bev_observed_mask_path", 1),
        (_corrupt_rotation_determinant, "camera2base", 1),
        (_corrupt_point_byte_length, "lidar_path", 1),
        (_corrupt_token_uniqueness, "token", 2),
        (_corrupt_split_overlap, "splits", 1),
    ),
)
def test_validator_names_the_field_for_corruptions(
    tmp_path, corrupt, field_name, frame_count
):
    build_dataset(tmp_path, frame_count=frame_count)
    corrupt(tmp_path)

    with pytest.raises(DatasetValidationError, match=rf"field={field_name}"):
        validate_dataset(tmp_path)


@pytest.mark.parametrize(
    "path_text",
    (
        "scene_a\\images\\000000.png",
        "scene_a/images/./000000.png",
        "images/./000000.png",
    ),
)
def test_validator_rejects_noncanonical_manifest_path_spelling(
    tmp_path, path_text
):
    build_dataset(tmp_path)
    _rewrite_manifest(
        tmp_path,
        lambda records: records[0].__setitem__("image_path", path_text),
    )

    with pytest.raises(
        DatasetValidationError, match=r"scene=scene_a.*field=image_path"
    ):
        validate_dataset(tmp_path)


def test_validator_accepts_canonical_scene_relative_manifest_paths(tmp_path):
    build_dataset(tmp_path)
    _rewrite_manifest(
        tmp_path,
        lambda records: records[0].__setitem__(
            "image_path", "images/000000.png"
        ),
    )

    assert validate_dataset(tmp_path).valid


def test_validator_rejects_scene_directory_symlink_escape(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(root)
    escaped_scene = tmp_path / "escaped_scene"
    (root / "scene_a").rename(escaped_scene)
    (root / "scene_a").symlink_to(escaped_scene, target_is_directory=True)

    with pytest.raises(
        DatasetValidationError,
        match=r"dataset=fixture_v3 scene=scene_a.*field=scene_directory",
    ):
        validate_dataset(root)


def test_validator_rejects_manifest_integer_pose_valid(tmp_path):
    build_dataset(tmp_path)
    _rewrite_manifest(
        tmp_path,
        lambda records: records[0].__setitem__("pose_valid", 1),
    )

    with pytest.raises(
        DatasetValidationError, match=r"scene=scene_a.*field=pose_valid"
    ):
        validate_dataset(tmp_path)


@pytest.mark.parametrize(
    "filename", ("scene_infos.pkl", "robot_infos_train.pkl")
)
def test_validator_rejects_local_index_integer_pose_valid(tmp_path, filename):
    build_dataset(tmp_path)

    def update(payload):
        payload["infos"][0]["pose_valid"] = 1

    _rewrite_pickle(tmp_path / "scene_a" / filename, update)

    with pytest.raises(
        DatasetValidationError,
        match=r"dataset=fixture_v3 scene=scene_a frame=0 field=pose_valid",
    ):
        validate_dataset(tmp_path)


def test_validator_reconciles_embedded_scene_summary_fields(tmp_path):
    build_dataset(tmp_path)
    _rewrite_json(
        tmp_path / "multi_scene_summary.json",
        lambda summary: summary["scene_summaries"][0].__setitem__(
            "frame_count", 99
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_a.*"
            r"field=multi_scene_summary\.scene_summaries\.frame_count"
        ),
    ):
        validate_dataset(tmp_path)


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    (
        ("point_count", {"min": 99, "max": 99, "mean": 99.0}),
        ("per_class_sums", {}),
        ("observed_sum", 99),
    ),
)
def test_validator_reconciles_embedded_scene_summary_aggregates(
    tmp_path, field_name, tampered_value
):
    build_dataset(tmp_path)
    _rewrite_json(
        tmp_path / "multi_scene_summary.json",
        lambda summary: summary["scene_summaries"][0].__setitem__(
            field_name, tampered_value
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_a.*"
            rf"field=multi_scene_summary\.scene_summaries\.{field_name}"
        ),
    ):
        validate_dataset(tmp_path)


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_validator_requires_exact_embedded_scene_summary_keys(
    tmp_path, mutation
):
    build_dataset(tmp_path)

    def update(summary):
        embedded = summary["scene_summaries"][0]
        if mutation == "extra":
            embedded["unexpected"] = 1
        else:
            embedded.pop("observed_sum")

    _rewrite_json(tmp_path / "multi_scene_summary.json", update)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_a.*"
            r"field=multi_scene_summary\.scene_summaries\.keys"
        ),
    ):
        validate_dataset(tmp_path)


def test_validator_rejects_duplicate_embedded_scene_summary(tmp_path):
    build_dataset(tmp_path)

    def duplicate(summary):
        summary["scene_summaries"].append(
            dict(summary["scene_summaries"][0])
        )

    _rewrite_json(tmp_path / "multi_scene_summary.json", duplicate)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_a.*"
            r"field=multi_scene_summary\.scene_summaries\.scene_id"
        ),
    ):
        validate_dataset(tmp_path)


def test_validator_rejects_missing_embedded_scene_summary(tmp_path):
    build_dataset(tmp_path)
    _rewrite_json(
        tmp_path / "multi_scene_summary.json",
        lambda summary: summary["scene_summaries"].clear(),
    )

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_a.*"
            r"field=multi_scene_summary\.scene_summaries\.scene_id"
        ),
    ):
        validate_dataset(tmp_path)


def test_validator_rejects_extra_embedded_scene_summary(tmp_path):
    build_dataset(tmp_path)

    def add_extra(summary):
        extra = dict(summary["scene_summaries"][0])
        extra["scene_id"] = "scene_extra"
        summary["scene_summaries"].append(extra)

    _rewrite_json(tmp_path / "multi_scene_summary.json", add_extra)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_extra.*"
            r"field=multi_scene_summary\.scene_summaries\.scene_id"
        ),
    ):
        validate_dataset(tmp_path)


@pytest.mark.parametrize("malformed_scene_id", ([], 7))
def test_validator_contextualizes_wrong_type_root_index_scene_id(
    tmp_path, malformed_scene_id
):
    build_dataset(tmp_path)

    def update(payload):
        payload["infos"][0]["scene_id"] = malformed_scene_id

    _rewrite_pickle(tmp_path / "robot_infos_train.pkl", update)

    with pytest.raises(
        DatasetValidationError,
        match=r"dataset=fixture_v3 .*frame=0 field=scene_id",
    ):
        validate_dataset(tmp_path)


def test_validator_contextualizes_wrong_type_root_index_path(tmp_path):
    build_dataset(tmp_path)

    def update(payload):
        payload["infos"][0]["image_path"] = np.array(
            ["scene_a/images/000000.png", "unexpected"]
        )

    _rewrite_pickle(tmp_path / "robot_infos_train.pkl", update)

    with pytest.raises(
        DatasetValidationError,
        match=r"dataset=fixture_v3 scene=scene_a frame=0 field=image_path",
    ):
        validate_dataset(tmp_path)


def test_validator_contextualizes_non_string_local_index_key(tmp_path):
    build_dataset(tmp_path)

    def update(payload):
        payload["infos"][0][1] = "invalid"

    _rewrite_pickle(tmp_path / "scene_a" / "scene_infos.pkl", update)

    with pytest.raises(
        DatasetValidationError,
        match=(
            r"dataset=fixture_v3 scene=scene_a frame=0 "
            r"field=scene_infos\.pkl\.infos\.keys"
        ),
    ):
        validate_dataset(tmp_path)


@pytest.mark.parametrize("field_name", ("dataset_id", "token"))
def test_validator_contextualizes_wrong_type_root_index_identity(
    tmp_path, field_name
):
    build_dataset(tmp_path)

    def update(payload):
        payload["infos"][0][field_name] = np.array(["invalid", "structure"])

    _rewrite_all_indexes(tmp_path, update)

    with pytest.raises(
        DatasetValidationError,
        match=rf"dataset=fixture_v3 scene=scene_a frame=0 field={field_name}",
    ):
        validate_dataset(tmp_path)


def test_validator_contextualizes_ragged_matrix(tmp_path):
    build_dataset(tmp_path)
    ragged = [[1.0, 0.0], [0.0]]

    def update(payload):
        payload["infos"][0]["camera2base"] = ragged

    _rewrite_all_indexes(tmp_path, update)
    _rewrite_manifest(
        tmp_path,
        lambda records: records[0].__setitem__("camera2base", ragged),
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"dataset=fixture_v3 scene=scene_a frame=0 field=camera2base",
    ):
        validate_dataset(tmp_path)
