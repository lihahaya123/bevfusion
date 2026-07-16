import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from .schema import (
    BEV_SHAPE,
    BEV_XBOUND,
    BEV_YBOUND,
    MAP_CLASSES,
    OBSERVED_MASK_SHAPE,
    POINT_DIMENSIONS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SchemaError,
    canonical_token,
    normalize_relative_path,
)


_SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_PATH_FIELDS = (
    "image_path",
    "lidar_path",
    "bev_mask_path",
    "bev_observed_mask_path",
)
_OPTIONAL_PATH_FIELDS = (
    "bev_supervision_mask_path",
    "depth_path",
    "semantic_path",
)
_PATH_FIELDS = _REQUIRED_PATH_FIELDS + _OPTIONAL_PATH_FIELDS
_POSE_FIELDS = (
    "cam_intrinsic",
    "camera2base",
    "lidar2base",
    "T_map_base",
    "pose_valid",
)


class DatasetValidationError(RuntimeError):
    def __init__(
        self,
        dataset_id: object,
        scene_id: object,
        frame_id: object,
        field_name: str,
        message: str,
    ) -> None:
        context = (
            f"dataset={dataset_id} scene={scene_id} frame={frame_id} "
            f"field={field_name}"
        )
        super().__init__(f"{context}: {message}")


@dataclass
class ValidationReport:
    valid: bool
    frame_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


@dataclass
class _FrameStatistics:
    scene_id: str
    frame_id: int
    point_count: int
    class_sums: np.ndarray
    observed_sum: int
    intensity_out_of_range: bool


def validate_dataset(root: Path, split: Optional[str] = None) -> ValidationReport:
    root = Path(root).expanduser().resolve()
    metadata = _load_root_metadata(root)
    splits = _load_and_validate_splits(root, metadata)
    if split is not None and split not in _SPLIT_NAMES:
        _fail(
            metadata,
            None,
            "split",
            f"expected one of {_SPLIT_NAMES}; actual={split!r}",
        )
    selected = (split,) if split is not None else _SPLIT_NAMES
    counts = {name: 0 for name in _SPLIT_NAMES}
    warnings: List[str] = []
    seen_tokens: Set[str] = set()

    for split_name in selected:
        infos = _load_index(root, split_name, metadata)
        expected_scenes = set(splits[split_name])
        manifests = _validate_manifests_match_index(
            root, metadata, split_name, expected_scenes, infos
        )
        previous_by_scene: Dict[str, Mapping[str, object]] = {}
        statistics: List[_FrameStatistics] = []
        for info in infos:
            _validate_frame_info(
                root, metadata, split_name, expected_scenes, info
            )
            token = str(info["token"])
            if token in seen_tokens:
                _fail(
                    metadata,
                    info,
                    "token",
                    f"expected a globally unique token; actual={token!r}",
                )
            seen_tokens.add(token)
            _validate_sequence(previous_by_scene, metadata, info)
            manifest_record = manifests[str(info["scene_id"])][
                int(info["frame_id"])
            ]
            statistics.append(
                _validate_artifacts(root, metadata, info, manifest_record)
            )
            counts[split_name] += 1
        _validate_split_summary(
            root, metadata, split_name, expected_scenes, infos
        )
        warnings.extend(
            _quality_warnings(metadata, split_name, statistics)
        )

    _validate_root_summary(root, metadata, selected, counts, splits)
    return ValidationReport(valid=True, frame_counts=counts, warnings=warnings)


def _load_root_metadata(root: Path) -> Dict[str, object]:
    unknown = {
        "dataset_id": "<unknown>",
    }
    if not root.is_dir():
        _fail(
            unknown,
            None,
            "root",
            f"expected an existing dataset directory; actual={root}",
        )
    path = root / "dataset_metadata.json"
    metadata = _load_json(path, unknown, None, "dataset_metadata")
    if not isinstance(metadata, dict):
        _fail(
            unknown,
            None,
            "dataset_metadata",
            f"expected a JSON object; actual={type(metadata).__name__}",
        )
    dataset_id = metadata.get("dataset_id", "<unknown>")
    context = {"dataset_id": dataset_id}
    if not isinstance(dataset_id, str) or not dataset_id:
        _fail(
            context,
            None,
            "dataset_id",
            f"expected a non-empty string; actual={dataset_id!r}",
        )
    expected = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "map_classes": list(MAP_CLASSES),
        "bev": {
            "xbound": list(BEV_XBOUND),
            "ybound": list(BEV_YBOUND),
            "shape": list(BEV_SHAPE),
            "encoding": "uint8_multihot",
            "observed_mask_shape": list(OBSERVED_MASK_SHAPE),
        },
        "points": {
            "dtype": "float32",
            "dimensions": list(POINT_DIMENSIONS),
        },
    }
    for field_name, expected_value in expected.items():
        actual = metadata.get(field_name)
        if actual != expected_value:
            _fail(
                context,
                None,
                field_name,
                f"expected={expected_value!r}; actual={actual!r}",
            )
    return metadata


def _load_and_validate_splits(
    root: Path, metadata: Mapping[str, object]
) -> Dict[str, List[str]]:
    splits = _load_json(root / "splits.json", metadata, None, "splits")
    if not isinstance(splits, dict) or set(splits) != set(_SPLIT_NAMES):
        actual = sorted(splits) if isinstance(splits, dict) else type(splits).__name__
        _fail(
            metadata,
            None,
            "splits",
            f"expected exactly {_SPLIT_NAMES}; actual={actual!r}",
        )
    owner: Dict[str, str] = {}
    normalized: Dict[str, List[str]] = {}
    for split_name in _SPLIT_NAMES:
        scene_ids = splits[split_name]
        if not isinstance(scene_ids, list):
            _fail(
                metadata,
                None,
                f"splits.{split_name}",
                f"expected a list; actual={type(scene_ids).__name__}",
            )
        normalized[split_name] = []
        for scene_id in scene_ids:
            if not isinstance(scene_id, str) or not scene_id:
                _fail(
                    metadata,
                    scene_id,
                    f"splits.{split_name}",
                    f"expected a non-empty scene ID; actual={scene_id!r}",
                )
            try:
                canonical = normalize_relative_path(scene_id)
                canonical_token(str(metadata["dataset_id"]), scene_id, 0)
            except SchemaError as error:
                _fail(
                    metadata,
                    scene_id,
                    f"splits.{split_name}",
                    f"expected a normalized root-relative scene ID; actual={scene_id!r} ({error})",
                )
            if canonical != scene_id:
                _fail(
                    metadata,
                    scene_id,
                    f"splits.{split_name}",
                    f"expected normalized POSIX form {canonical!r}; actual={scene_id!r}",
                )
            previous = owner.setdefault(scene_id, split_name)
            if previous != split_name:
                _fail(
                    metadata,
                    scene_id,
                    "splits",
                    f"expected disjoint scene sets; actual={scene_id!r} in {previous!r} and {split_name!r}",
                )
            if scene_id in normalized[split_name]:
                _fail(
                    metadata,
                    scene_id,
                    f"splits.{split_name}",
                    f"expected unique scene IDs; actual duplicate={scene_id!r}",
                )
            normalized[split_name].append(scene_id)
    return normalized


def _load_index(
    root: Path, split_name: str, metadata: Mapping[str, object]
) -> List[Dict[str, object]]:
    path = root / f"robot_infos_{split_name}.pkl"
    payload = _load_pickle(path, metadata, None, f"robot_infos_{split_name}")
    _validate_index_payload_metadata(
        metadata, payload, f"robot_infos_{split_name}", None
    )
    infos = payload.get("infos")
    if not isinstance(infos, list) or not all(
        isinstance(info, dict) for info in infos
    ):
        _fail(
            metadata,
            None,
            f"robot_infos_{split_name}.infos",
            f"expected a list of objects; actual={type(infos).__name__}",
        )
    return infos


def _validate_index_payload_metadata(
    metadata: Mapping[str, object],
    payload: object,
    field_name: str,
    expected_scene_split: Optional[str],
) -> None:
    if not isinstance(payload, dict):
        _fail(
            metadata,
            None,
            field_name,
            f"expected a pickle object; actual={type(payload).__name__}",
        )
    index_metadata = payload.get("metadata")
    if not isinstance(index_metadata, dict):
        _fail(
            metadata,
            None,
            f"{field_name}.metadata",
            f"expected a metadata object; actual={type(index_metadata).__name__}",
        )
    comparable = dict(index_metadata)
    actual_scene_split = comparable.pop("scene_split", None)
    if expected_scene_split is None:
        if actual_scene_split is not None:
            _fail(
                metadata,
                None,
                f"{field_name}.metadata.scene_split",
                f"expected=None; actual={actual_scene_split!r}",
            )
    elif actual_scene_split != expected_scene_split:
        _fail(
            metadata,
            None,
            f"{field_name}.metadata.scene_split",
            f"expected={expected_scene_split!r}; actual={actual_scene_split!r}",
        )
    if not _values_equal(comparable, dict(metadata)):
        _fail(
            metadata,
            None,
            f"{field_name}.metadata",
            "expected metadata identical to dataset_metadata.json; actual differs",
        )


def _validated_scene_directory(
    root: Path, metadata: Mapping[str, object], scene_id: str
) -> Path:
    candidate = root / scene_id
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail(
            metadata,
            scene_id,
            "scene_directory",
            f"expected an existing resolvable directory inside {root}; actual={candidate} ({error})",
        )
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(
            metadata,
            scene_id,
            "scene_directory",
            f"expected path to resolve inside {root}; actual={resolved}",
        )
    if not resolved.is_dir():
        _fail(
            metadata,
            scene_id,
            "scene_directory",
            f"expected a directory; actual={resolved}",
        )
    return resolved


def _validate_manifests_match_index(
    root: Path,
    metadata: Mapping[str, object],
    split_name: str,
    expected_scenes: Set[str],
    infos: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[int, Dict[str, object]]]:
    infos_by_scene: Dict[str, List[Mapping[str, object]]] = {
        scene_id: [] for scene_id in expected_scenes
    }
    for info in infos:
        scene_id = info.get("scene_id")
        if not isinstance(scene_id, str):
            _fail(
                metadata,
                info,
                "scene_id",
                f"expected a string scene ID; actual={scene_id!r}",
            )
        if scene_id not in expected_scenes:
            _fail(
                metadata,
                info,
                "scene_id",
                f"expected a scene from split {split_name!r}: {sorted(expected_scenes)!r}; actual={scene_id!r}",
            )
        infos_by_scene[scene_id].append(info)

    manifest_maps: Dict[str, Dict[int, Dict[str, object]]] = {}
    for scene_id in sorted(expected_scenes):
        scene_dir = _validated_scene_directory(root, metadata, scene_id)
        records = _load_manifest(scene_dir, metadata, scene_id)
        by_frame: Dict[int, Dict[str, object]] = {}
        for record in records:
            _validate_manifest_record(metadata, scene_id, record)
            frame_id = int(record["frame_id"])
            if frame_id in by_frame:
                _fail(
                    metadata,
                    record,
                    "frame_id",
                    f"expected unique manifest frame IDs; actual duplicate={frame_id}",
                )
            normalized_record = dict(record)
            for path_field in _PATH_FIELDS:
                value = record.get(path_field)
                if value is not None:
                    normalized_record[path_field] = _manifest_path_to_root(
                        metadata, scene_id, frame_id, path_field, value
                    )
            by_frame[frame_id] = normalized_record

        root_infos = infos_by_scene[scene_id]
        root_by_frame: Dict[int, Mapping[str, object]] = {}
        for info in root_infos:
            frame_id = _integer_field(metadata, info, "frame_id")
            if frame_id in root_by_frame:
                _fail(
                    metadata,
                    info,
                    "frame_id",
                    f"expected unique root-index frame IDs; actual duplicate={frame_id}",
                )
            root_by_frame[frame_id] = info
        if set(by_frame) != set(root_by_frame):
            _fail(
                metadata,
                scene_id,
                "frame_count",
                f"expected manifest frame IDs {sorted(by_frame)!r}; actual root-index frame IDs {sorted(root_by_frame)!r}",
            )
        for frame_id, record in by_frame.items():
            _compare_manifest_and_info(
                metadata, scene_id, frame_id, record, root_by_frame[frame_id]
            )
        _validate_scene_indexes(
            scene_dir, metadata, split_name, scene_id, root_infos
        )
        _validate_scene_artifact_sets(root, metadata, scene_id, by_frame)
        manifest_maps[scene_id] = by_frame

    return manifest_maps


def _load_manifest(
    scene_dir: Path, metadata: Mapping[str, object], scene_id: str
) -> List[Dict[str, object]]:
    path = scene_dir / "manifest.jsonl"
    if not path.is_file():
        _fail(
            metadata,
            scene_id,
            "manifest",
            f"expected an existing scene manifest; actual missing={path}",
        )
    records: List[Dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(
                        metadata,
                        scene_id,
                        "manifest",
                        f"expected one JSON object per nonblank line; actual blank line={line_number}",
                    )
                record = json.loads(line)
                if not isinstance(record, dict):
                    _fail(
                        metadata,
                        scene_id,
                        "manifest",
                        f"expected a JSON object at line {line_number}; actual={type(record).__name__}",
                    )
                records.append(record)
    except DatasetValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(
            metadata,
            scene_id,
            "manifest",
            f"expected readable JSON Lines; actual error={error}",
        )
    return records


def _validate_manifest_record(
    metadata: Mapping[str, object], scene_id: str, record: Mapping[str, object]
) -> None:
    frame_id = _integer_field(metadata, record, "frame_id")
    if record.get("dataset_id") != metadata["dataset_id"]:
        _fail(
            metadata,
            record,
            "dataset_id",
            f"expected={metadata['dataset_id']!r}; actual={record.get('dataset_id')!r}",
        )
    if record.get("scene_id") != scene_id:
        _fail(
            metadata,
            record,
            "scene_id",
            f"expected={scene_id!r}; actual={record.get('scene_id')!r}",
        )
    if record.get("pose_valid") is not True:
        _fail(
            metadata,
            record,
            "pose_valid",
            f"expected=True (JSON boolean); actual={record.get('pose_valid')!r}",
        )
    _integer_field(metadata, record, "timestamp")
    for path_field in _REQUIRED_PATH_FIELDS:
        if not isinstance(record.get(path_field), str):
            _fail(
                metadata,
                record,
                path_field,
                f"expected a non-empty relative path; actual={record.get(path_field)!r}",
            )
    for path_field in _OPTIONAL_PATH_FIELDS:
        value = record.get(path_field)
        if value is not None and not isinstance(value, str):
            _fail(
                metadata,
                record,
                path_field,
                f"expected a relative path or None; actual={value!r}",
            )
    if frame_id < 0:
        _fail(
            metadata,
            record,
            "frame_id",
            f"expected a non-negative integer; actual={frame_id}",
        )


def _manifest_path_to_root(
    metadata: Mapping[str, object],
    scene_id: str,
    frame_id: int,
    field_name: str,
    value: object,
) -> str:
    context = {"scene_id": scene_id, "frame_id": frame_id}
    if not isinstance(value, str):
        _fail(
            metadata,
            context,
            field_name,
            f"expected a canonical POSIX relative path; actual={value!r}",
        )
    try:
        normalized = normalize_relative_path(value)
    except SchemaError as error:
        _fail(
            metadata,
            context,
            field_name,
            f"expected a root-contained relative path; actual={value!r} ({error})",
        )
    if normalized != value:
        _fail(
            metadata,
            context,
            field_name,
            f"expected canonical POSIX spelling={normalized!r}; actual={value!r}",
        )
    scene_parts = PurePosixPath(scene_id).parts
    path_parts = PurePosixPath(normalized).parts
    if path_parts[: len(scene_parts)] != scene_parts:
        normalized = normalize_relative_path(
            PurePosixPath(scene_id) / normalized
        )
    return normalized


def _compare_manifest_and_info(
    metadata: Mapping[str, object],
    scene_id: str,
    frame_id: int,
    record: Mapping[str, object],
    info: Mapping[str, object],
) -> None:
    context = {"scene_id": scene_id, "frame_id": frame_id}
    for field_name in ("frame_id", "timestamp") + _PATH_FIELDS:
        expected = record.get(field_name)
        actual = info.get(field_name)
        if not _values_equal(expected, actual):
            _fail(
                metadata,
                context,
                field_name,
                f"expected manifest value={expected!r}; actual root-index value={actual!r}",
            )
    for field_name in _POSE_FIELDS + ("class_validity",):
        if field_name not in record or field_name not in info:
            _fail(
                metadata,
                context,
                field_name,
                f"expected field in manifest and root index; actual manifest={field_name in record}, root_index={field_name in info}",
            )
        if not _values_equal(record[field_name], info[field_name]):
            _fail(
                metadata,
                context,
                field_name,
                "expected manifest and root-index values to match; actual values differ",
            )


def _validate_scene_indexes(
    scene_dir: Path,
    metadata: Mapping[str, object],
    split_name: str,
    scene_id: str,
    root_infos: Sequence[Mapping[str, object]],
) -> None:
    for filename in ("scene_infos.pkl", f"robot_infos_{split_name}.pkl"):
        path = scene_dir / filename
        payload = _load_pickle(path, metadata, scene_id, filename)
        _validate_index_payload_metadata(
            metadata, payload, filename, split_name
        )
        local_infos = payload.get("infos")
        if not isinstance(local_infos, list):
            _fail(
                metadata,
                scene_id,
                f"{filename}.infos",
                f"expected a list; actual={type(local_infos).__name__}",
            )
        if len(local_infos) != len(root_infos):
            _fail(
                metadata,
                scene_id,
                "frame_count",
                f"expected {filename} count={len(local_infos)}; actual root-index count={len(root_infos)}",
            )
        for local_info, root_info in zip(local_infos, root_infos):
            if not isinstance(local_info, dict):
                _fail(
                    metadata,
                    scene_id,
                    f"{filename}.infos",
                    f"expected index records to be objects; actual={type(local_info).__name__}",
                )
            for source_name, index_info in (
                (filename, local_info),
                ("root_index", root_info),
            ):
                non_string_keys = [
                    key for key in index_info if not isinstance(key, str)
                ]
                if non_string_keys:
                    _fail(
                        metadata,
                        index_info,
                        f"{source_name}.infos.keys",
                        f"expected string field names; actual non-string keys={non_string_keys!r}",
                    )
            if local_info.get("pose_valid") is not True:
                _fail(
                    metadata,
                    local_info,
                    "pose_valid",
                    f"expected=True; actual={local_info.get('pose_valid')!r}",
                )
            keys = set(local_info).union(root_info)
            for field_name in sorted(keys):
                expected = local_info.get(field_name)
                actual = root_info.get(field_name)
                if not _values_equal(expected, actual):
                    _fail(
                        metadata,
                        root_info,
                        field_name,
                        f"expected {filename} value={expected!r}; actual root-index value={actual!r}",
                    )


def _validate_scene_artifact_sets(
    root: Path,
    metadata: Mapping[str, object],
    scene_id: str,
    records: Mapping[int, Mapping[str, object]],
) -> None:
    layouts = {
        "image_path": ("images", ".png"),
        "lidar_path": ("points", ".bin"),
        "bev_mask_path": ("bev_masks", ".npy"),
        "bev_observed_mask_path": ("bev_observed_masks", ".npy"),
        "bev_supervision_mask_path": ("bev_supervision_masks", ".npy"),
        "depth_path": ("depths", ".png"),
        "semantic_path": ("semantics", ".png"),
    }
    for field_name, (directory, suffix) in layouts.items():
        expected = {
            str(record[field_name])
            for record in records.values()
            if record.get(field_name) is not None
        }
        artifact_dir = root / scene_id / directory
        actual = (
            {
                path.relative_to(root).as_posix()
                for path in artifact_dir.iterdir()
                if path.is_file() and path.suffix == suffix
            }
            if artifact_dir.is_dir()
            else set()
        )
        if actual != expected:
            _fail(
                metadata,
                scene_id,
                field_name,
                f"expected artifact set={sorted(expected)!r}; actual={sorted(actual)!r}",
            )


def _validate_frame_info(
    root: Path,
    metadata: Mapping[str, object],
    split_name: str,
    expected_scenes: Set[str],
    info: Mapping[str, object],
) -> None:
    dataset_id = metadata["dataset_id"]
    actual_dataset_id = info.get("dataset_id")
    if not isinstance(actual_dataset_id, str):
        _fail(
            metadata,
            info,
            "dataset_id",
            f"expected a string dataset ID; actual={actual_dataset_id!r}",
        )
    if actual_dataset_id != dataset_id:
        _fail(
            metadata,
            info,
            "dataset_id",
            f"expected={dataset_id!r}; actual={actual_dataset_id!r}",
        )
    scene_id = info.get("scene_id")
    if not isinstance(scene_id, str):
        _fail(
            metadata,
            info,
            "scene_id",
            f"expected a string scene ID; actual={scene_id!r}",
        )
    if scene_id not in expected_scenes:
        _fail(
            metadata,
            info,
            "scene_id",
            f"expected a scene in split {split_name!r}; actual={scene_id!r}",
        )
    frame_id = _integer_field(metadata, info, "frame_id")
    if frame_id < 0:
        _fail(
            metadata,
            info,
            "frame_id",
            f"expected a non-negative integer; actual={frame_id}",
        )
    _integer_field(metadata, info, "timestamp")
    expected_token = canonical_token(str(dataset_id), str(scene_id), frame_id)
    token = info.get("token")
    if not isinstance(token, str):
        _fail(
            metadata,
            info,
            "token",
            f"expected a string token; actual={token!r}",
        )
    if token != expected_token:
        _fail(
            metadata,
            info,
            "token",
            f"expected={expected_token!r}; actual={token!r}",
        )
    if not isinstance(info.get("prev_token"), str):
        _fail(
            metadata,
            info,
            "prev_token",
            f"expected a string; actual={info.get('prev_token')!r}",
        )
    for field_name in _REQUIRED_PATH_FIELDS:
        _validated_artifact_path(root, metadata, info, field_name, required=True)
    for field_name in _OPTIONAL_PATH_FIELDS:
        _validated_artifact_path(root, metadata, info, field_name, required=False)
    _binary_array(
        metadata,
        info,
        "class_validity",
        _coerce_array(
            metadata, info, "class_validity", info.get("class_validity")
        ),
        (len(MAP_CLASSES),),
    )
    _validate_intrinsic_metadata(metadata, info)
    for field_name in ("camera2base", "lidar2base", "T_map_base"):
        _validate_transform(metadata, info, field_name)
    if info.get("pose_valid") is not True:
        _fail(
            metadata,
            info,
            "pose_valid",
            f"expected=True; actual={info.get('pose_valid')!r}",
        )


def _validate_sequence(
    previous_by_scene: Dict[str, Mapping[str, object]],
    metadata: Mapping[str, object],
    info: Mapping[str, object],
) -> None:
    scene_id = str(info["scene_id"])
    frame_id = int(info["frame_id"])
    previous = previous_by_scene.get(scene_id)
    if previous is None:
        if frame_id != 0:
            _fail(
                metadata,
                info,
                "frame_id",
                f"expected first frame_id=0; actual={frame_id}",
            )
        if info["prev_token"] != "":
            _fail(
                metadata,
                info,
                "prev_token",
                f"expected='' for the first scene frame; actual={info['prev_token']!r}",
            )
    else:
        expected_frame_id = int(previous["frame_id"]) + 1
        if frame_id != expected_frame_id:
            _fail(
                metadata,
                info,
                "frame_id",
                f"expected contiguous frame_id={expected_frame_id}; actual={frame_id}",
            )
        expected_prev = previous["token"]
        if info["prev_token"] != expected_prev:
            _fail(
                metadata,
                info,
                "prev_token",
                f"expected previous token={expected_prev!r}; actual={info['prev_token']!r}",
            )
        if int(info["timestamp"]) <= int(previous["timestamp"]):
            _fail(
                metadata,
                info,
                "timestamp",
                f"expected > {previous['timestamp']}; actual={info['timestamp']}",
            )
    previous_by_scene[scene_id] = info


def _validate_artifacts(
    root: Path,
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    manifest_record: Mapping[str, object],
) -> _FrameStatistics:
    image_path = _validated_artifact_path(
        root, metadata, info, "image_path", required=True
    )
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.mode != "RGB":
                _fail(
                    metadata,
                    info,
                    "image_path",
                    f"expected RGB PNG; actual mode={image.mode!r}",
                )
            width, height = image.size
    except DatasetValidationError:
        raise
    except (OSError, ValueError) as error:
        _fail(
            metadata,
            info,
            "image_path",
            f"expected a readable RGB PNG; actual error={error}",
        )

    intrinsic = _coerce_array(
        metadata, info, "cam_intrinsic", info["cam_intrinsic"]
    )
    _validate_intrinsic_for_image(metadata, info, intrinsic, width, height)

    labels = _load_npy(
        _validated_artifact_path(
            root, metadata, info, "bev_mask_path", required=True
        ),
        metadata,
        info,
        "bev_mask_path",
    )
    _binary_array(metadata, info, "bev_mask_path", labels, BEV_SHAPE)
    observed = _load_npy(
        _validated_artifact_path(
            root, metadata, info, "bev_observed_mask_path", required=True
        ),
        metadata,
        info,
        "bev_observed_mask_path",
    )
    _binary_array(
        metadata,
        info,
        "bev_observed_mask_path",
        observed,
        OBSERVED_MASK_SHAPE,
    )
    supervision_path = _validated_artifact_path(
        root,
        metadata,
        info,
        "bev_supervision_mask_path",
        required=False,
    )
    if supervision_path is not None:
        supervision = _load_npy(
            supervision_path,
            metadata,
            info,
            "bev_supervision_mask_path",
        )
        _binary_array(
            metadata,
            info,
            "bev_supervision_mask_path",
            supervision,
            BEV_SHAPE,
        )
    if np.any(labels[:, observed == 0]):
        actual = int(np.count_nonzero(labels[:, observed == 0]))
        _fail(
            metadata,
            info,
            "bev_mask_path",
            f"expected zero labels outside observed cells; actual nonzero_count={actual}",
        )

    for field_name in ("depth_path", "semantic_path"):
        optional_path = _validated_artifact_path(
            root, metadata, info, field_name, required=False
        )
        if optional_path is not None:
            _validate_uint16_png(
                optional_path,
                metadata,
                info,
                field_name,
                (height, width),
            )

    points_path = _validated_artifact_path(
        root, metadata, info, "lidar_path", required=True
    )
    byte_count = points_path.stat().st_size
    bytes_per_point = len(POINT_DIMENSIONS) * np.dtype(np.float32).itemsize
    if byte_count % bytes_per_point:
        _fail(
            metadata,
            info,
            "lidar_path",
            f"expected byte length divisible by {bytes_per_point}; actual={byte_count}",
        )
    try:
        points = np.fromfile(str(points_path), dtype=np.float32).reshape(
            -1, len(POINT_DIMENSIONS)
        )
    except (OSError, ValueError) as error:
        _fail(
            metadata,
            info,
            "lidar_path",
            f"expected readable float32 point records; actual error={error}",
        )
    if not np.isfinite(points).all():
        _fail(
            metadata,
            info,
            "lidar_path",
            "expected all finite float32 point values; actual contains non-finite values",
        )

    point_count = int(points.shape[0])
    class_sums = labels.sum(axis=(1, 2), dtype=np.int64)
    observed_sum = int(observed.sum(dtype=np.int64))
    for field_name, expected, actual in (
        ("point_count", point_count, manifest_record.get("point_count")),
        (
            "per_class_sums",
            class_sums.tolist(),
            manifest_record.get("per_class_sums"),
        ),
        ("observed_sum", observed_sum, manifest_record.get("observed_sum")),
    ):
        if expected != actual:
            _fail(
                metadata,
                info,
                field_name,
                f"expected artifact-derived value={expected!r}; actual manifest value={actual!r}",
            )

    return _FrameStatistics(
        scene_id=str(info["scene_id"]),
        frame_id=int(info["frame_id"]),
        point_count=point_count,
        class_sums=class_sums,
        observed_sum=observed_sum,
        intensity_out_of_range=bool(
            points.size
            and np.any((points[:, 3] < 0.0) | (points[:, 3] > 1.0))
        ),
    )


def _validate_intrinsic_metadata(
    metadata: Mapping[str, object], info: Mapping[str, object]
) -> None:
    intrinsic = _float32_array(
        metadata, info, "cam_intrinsic", (3, 3)
    )
    if not np.allclose(
        intrinsic[2], np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6
    ):
        _fail(
            metadata,
            info,
            "cam_intrinsic",
            f"expected homogeneous bottom row [0, 0, 1]; actual={intrinsic[2].tolist()!r}",
        )
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        _fail(
            metadata,
            info,
            "cam_intrinsic",
            f"expected positive focal lengths; actual fx={intrinsic[0, 0]}, fy={intrinsic[1, 1]}",
        )


def _validate_intrinsic_for_image(
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> None:
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        _fail(
            metadata,
            info,
            "cam_intrinsic",
            f"expected principal point inside image width={width}, height={height}; actual cx={cx}, cy={cy}",
        )


def _validate_transform(
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
) -> None:
    matrix = _float32_array(metadata, info, field_name, (4, 4))
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(matrix[3], expected_bottom, atol=1e-6):
        _fail(
            metadata,
            info,
            field_name,
            f"expected homogeneous bottom row {expected_bottom.tolist()!r}; actual={matrix[3].tolist()!r}",
        )
    rotation = matrix[:3, :3]
    gram = rotation.T @ rotation
    if not np.allclose(gram, np.eye(3, dtype=np.float32), atol=1e-3):
        _fail(
            metadata,
            info,
            field_name,
            f"expected an orthonormal rotation within 1e-3; actual R^T R={gram.tolist()!r}",
        )
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-3:
        _fail(
            metadata,
            info,
            field_name,
            f"expected rotation determinant within 1e-3 of +1; actual={determinant}",
        )


def _float32_array(
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
    shape: Tuple[int, ...],
) -> np.ndarray:
    value = info.get(field_name)
    array = _coerce_array(metadata, info, field_name, value)
    if array.shape != shape:
        _fail(
            metadata,
            info,
            field_name,
            f"expected shape={shape}; actual={array.shape}",
        )
    if array.dtype != np.float32:
        _fail(
            metadata,
            info,
            field_name,
            f"expected dtype=float32; actual={array.dtype}",
        )
    if not np.isfinite(array).all():
        _fail(
            metadata,
            info,
            field_name,
            "expected all finite values; actual contains non-finite values",
        )
    return array


def _coerce_array(
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
    value: object,
) -> np.ndarray:
    try:
        return np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        _fail(
            metadata,
            info,
            field_name,
            f"expected a regular numeric array; actual={value!r} ({error})",
        )


def _binary_array(
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
    array: np.ndarray,
    shape: Sequence[int],
) -> None:
    if array.shape != tuple(shape):
        _fail(
            metadata,
            info,
            field_name,
            f"expected shape={tuple(shape)}; actual={array.shape}",
        )
    if array.dtype != np.uint8:
        _fail(
            metadata,
            info,
            field_name,
            f"expected dtype=uint8; actual={array.dtype}",
        )
    if not np.isin(array, (0, 1)).all():
        values = np.unique(array).tolist()
        _fail(
            metadata,
            info,
            field_name,
            f"expected binary values in [0, 1]; actual unique values={values!r}",
        )


def _validated_artifact_path(
    root: Path,
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
    required: bool,
) -> Optional[Path]:
    value = info.get(field_name)
    if value is None:
        if required:
            _fail(
                metadata,
                info,
                field_name,
                "expected a root-contained artifact path; actual=None",
            )
        return None
    if not isinstance(value, str):
        _fail(
            metadata,
            info,
            field_name,
            f"expected a string path; actual={value!r}",
        )
    try:
        normalized = normalize_relative_path(value)
    except SchemaError as error:
        _fail(
            metadata,
            info,
            field_name,
            f"expected a root-contained relative path; actual={value!r} ({error})",
        )
    if normalized != value:
        _fail(
            metadata,
            info,
            field_name,
            f"expected normalized POSIX path={normalized!r}; actual={value!r}",
        )
    candidate = root / normalized
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(
            metadata,
            info,
            field_name,
            f"expected path to resolve inside {root}; actual={resolved}",
        )
    if not resolved.is_file():
        _fail(
            metadata,
            info,
            field_name,
            f"expected an existing file; actual={resolved}",
        )
    return resolved


def _load_npy(
    path: Path,
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
) -> np.ndarray:
    try:
        array = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as error:
        _fail(
            metadata,
            info,
            field_name,
            f"expected a readable NumPy array; actual error={error}",
        )
    if not isinstance(array, np.ndarray):
        _fail(
            metadata,
            info,
            field_name,
            f"expected one NumPy array; actual={type(array).__name__}",
        )
    return array


def _validate_uint16_png(
    path: Path,
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
    expected_shape: Tuple[int, int],
) -> None:
    try:
        header = path.read_bytes()[:26]
        with Image.open(path) as image:
            image.load()
            actual_shape = (image.height, image.width)
    except (OSError, ValueError) as error:
        _fail(
            metadata,
            info,
            field_name,
            f"expected a readable uint16 grayscale PNG; actual error={error}",
        )
    is_uint16_grayscale = (
        len(header) >= 26
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[24] == 16
        and header[25] == 0
    )
    if not is_uint16_grayscale:
        bit_depth = header[24] if len(header) > 24 else None
        color_type = header[25] if len(header) > 25 else None
        _fail(
            metadata,
            info,
            field_name,
            f"expected uint16 grayscale PNG encoding; actual bit_depth={bit_depth}, color_type={color_type}",
        )
    if actual_shape != expected_shape:
        _fail(
            metadata,
            info,
            field_name,
            f"expected shape={expected_shape}; actual={actual_shape}",
        )


def _validate_split_summary(
    root: Path,
    metadata: Mapping[str, object],
    split_name: str,
    expected_scenes: Set[str],
    infos: Sequence[Mapping[str, object]],
) -> None:
    counts: Dict[str, int] = {scene_id: 0 for scene_id in expected_scenes}
    for info in infos:
        counts[str(info["scene_id"])] += 1
    for scene_id in expected_scenes:
        scene_dir = _validated_scene_directory(root, metadata, scene_id)
        summary = _load_json(
            scene_dir / "summary.json",
            metadata,
            scene_id,
            "summary",
        )
        if not isinstance(summary, dict):
            _fail(
                metadata,
                scene_id,
                "summary",
                f"expected a JSON object; actual={type(summary).__name__}",
            )
        expected_fields = {
            "status": "complete",
            "scene_id": scene_id,
            "split": split_name,
            "generation_fingerprint": metadata.get("generation_fingerprint"),
            "frame_count": counts[scene_id],
        }
        for field_name, expected in expected_fields.items():
            actual = summary.get(field_name)
            if actual != expected:
                _fail(
                    metadata,
                    scene_id,
                    f"summary.{field_name}",
                    f"expected={expected!r}; actual={actual!r}",
                )


def _validate_root_summary(
    root: Path,
    metadata: Mapping[str, object],
    selected: Sequence[str],
    counts: Mapping[str, int],
    splits: Mapping[str, Sequence[str]],
) -> None:
    summary = _load_json(
        root / "multi_scene_summary.json",
        metadata,
        None,
        "multi_scene_summary",
    )
    if not isinstance(summary, dict):
        _fail(
            metadata,
            None,
            "multi_scene_summary",
            f"expected a JSON object; actual={type(summary).__name__}",
        )
    if summary.get("status") != "complete":
        _fail(
            metadata,
            None,
            "multi_scene_summary.status",
            f"expected='complete'; actual={summary.get('status')!r}",
        )
    info_counts = summary.get("info_counts")
    if not isinstance(info_counts, dict):
        _fail(
            metadata,
            None,
            "multi_scene_summary.info_counts",
            f"expected an object; actual={type(info_counts).__name__}",
        )
    for split_name in selected:
        if info_counts.get(split_name) != counts[split_name]:
            _fail(
                metadata,
                None,
                f"multi_scene_summary.info_counts.{split_name}",
                f"expected={counts[split_name]}; actual={info_counts.get(split_name)!r}",
            )
    scene_summaries = summary.get("scene_summaries")
    if not isinstance(scene_summaries, list):
        _fail(
            metadata,
            None,
            "multi_scene_summary.scene_summaries",
            f"expected a list; actual={type(scene_summaries).__name__}",
        )
    expected_scene_splits = {
        scene_id: split_name
        for split_name in _SPLIT_NAMES
        for scene_id in splits[split_name]
    }
    embedded_by_scene: Dict[str, Mapping[str, object]] = {}
    for item in scene_summaries:
        if not isinstance(item, dict):
            _fail(
                metadata,
                None,
                "multi_scene_summary.scene_summaries",
                f"expected summary objects; actual item={item!r}",
            )
        scene_id = item.get("scene_id")
        if not isinstance(scene_id, str):
            _fail(
                metadata,
                item,
                "multi_scene_summary.scene_summaries.scene_id",
                f"expected a string scene ID; actual={scene_id!r}",
            )
        if scene_id not in expected_scene_splits:
            _fail(
                metadata,
                item,
                "multi_scene_summary.scene_summaries.scene_id",
                f"expected one of {sorted(expected_scene_splits)!r}; actual={scene_id!r}",
            )
        if scene_id in embedded_by_scene:
            _fail(
                metadata,
                item,
                "multi_scene_summary.scene_summaries.scene_id",
                f"expected one summary per scene; actual duplicate={scene_id!r}",
            )
        embedded_by_scene[scene_id] = item

    missing = set(expected_scene_splits).difference(embedded_by_scene)
    if missing:
        scene_id = sorted(missing)[0]
        _fail(
            metadata,
            scene_id,
            "multi_scene_summary.scene_summaries.scene_id",
            f"expected embedded summary for every scene; actual missing={sorted(missing)!r}",
        )

    for scene_id, split_name in expected_scene_splits.items():
        scene_dir = _validated_scene_directory(root, metadata, scene_id)
        per_scene = _load_json(
            scene_dir / "summary.json",
            metadata,
            scene_id,
            "summary",
        )
        if not isinstance(per_scene, dict):
            _fail(
                metadata,
                scene_id,
                "summary",
                f"expected a JSON object; actual={type(per_scene).__name__}",
            )
        embedded = embedded_by_scene[scene_id]
        expected_keys = set(per_scene)
        actual_keys = set(embedded)
        if actual_keys != expected_keys:
            _fail(
                metadata,
                {"scene_id": scene_id},
                "multi_scene_summary.scene_summaries.keys",
                f"expected per-scene keys={sorted(expected_keys)!r}; actual embedded keys={sorted(actual_keys)!r}",
            )
        for field_name in sorted(expected_keys):
            expected = per_scene[field_name]
            actual = embedded[field_name]
            if not _values_equal(actual, expected):
                _fail(
                    metadata,
                    {"scene_id": scene_id},
                    f"multi_scene_summary.scene_summaries.{field_name}",
                    f"expected per-scene value={expected!r}; actual embedded value={actual!r}",
                )
        if per_scene.get("split") != split_name:
            _fail(
                metadata,
                scene_id,
                "summary.split",
                f"expected={split_name!r}; actual={per_scene.get('split')!r}",
            )


def _quality_warnings(
    metadata: Mapping[str, object],
    split_name: str,
    statistics: Sequence[_FrameStatistics],
) -> List[str]:
    if not statistics:
        return []
    warnings: List[str] = []
    totals = np.sum(
        np.stack([item.class_sums for item in statistics], axis=0), axis=0
    )
    for class_name, total in zip(MAP_CLASSES, totals.tolist()):
        if total == 0:
            warnings.append(
                f"dataset={metadata['dataset_id']} split={split_name} class={class_name}: zero class total"
            )
    observed_coverage = sum(
        item.observed_sum for item in statistics
    ) / float(len(statistics) * OBSERVED_MASK_SHAPE[0] * OBSERVED_MASK_SHAPE[1])
    if observed_coverage < 0.01:
        warnings.append(
            f"dataset={metadata['dataset_id']} split={split_name}: observed coverage {observed_coverage:.6f} is below 1%"
        )
    point_counts = np.asarray(
        [item.point_count for item in statistics], dtype=np.float64
    )
    median = float(np.median(point_counts))
    for item in statistics:
        is_outlier = (
            item.point_count > 0 if median == 0 else (
                item.point_count < median / 5.0
                or item.point_count > median * 5.0
            )
        )
        if is_outlier:
            warnings.append(
                f"dataset={metadata['dataset_id']} split={split_name} scene={item.scene_id} frame={item.frame_id}: point_count={item.point_count} outside five times median={median:g}"
            )
        if item.intensity_out_of_range:
            warnings.append(
                f"dataset={metadata['dataset_id']} split={split_name} scene={item.scene_id} frame={item.frame_id}: intensity outside [0,1]"
            )
    return warnings


def _load_json(
    path: Path,
    metadata: Mapping[str, object],
    scene_id: Optional[str],
    field_name: str,
) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(
            metadata,
            scene_id,
            field_name,
            f"expected readable JSON at {path}; actual error={error}",
        )


def _load_pickle(
    path: Path,
    metadata: Mapping[str, object],
    scene_id: Optional[str],
    field_name: str,
) -> object:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as error:
        _fail(
            metadata,
            scene_id,
            field_name,
            f"expected readable pickle at {path}; actual error={error}",
        )


def _integer_field(
    metadata: Mapping[str, object],
    info: Mapping[str, object],
    field_name: str,
) -> int:
    value = info.get(field_name)
    if not isinstance(value, (int, np.integer)) or isinstance(
        value, (bool, np.bool_)
    ):
        _fail(
            metadata,
            info,
            field_name,
            f"expected an integer; actual={value!r}",
        )
    return int(value)


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _fail(
    metadata: Mapping[str, object],
    info: object,
    field_name: str,
    message: str,
) -> None:
    dataset_id = metadata.get("dataset_id", "<unknown>")
    if isinstance(info, Mapping):
        scene_id = info.get("scene_id", "<root>")
        frame_id = info.get("frame_id", "<unknown>")
    elif isinstance(info, str):
        scene_id = info
        frame_id = "<all>"
    elif info is None:
        scene_id = "<root>"
        frame_id = "<all>"
    else:
        scene_id = info
        frame_id = "<unknown>"
    raise DatasetValidationError(
        dataset_id, scene_id, frame_id, field_name, message
    )


__all__ = (
    "DatasetValidationError",
    "ValidationReport",
    "validate_dataset",
)
