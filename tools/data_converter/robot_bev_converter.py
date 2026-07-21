"""Convert canonical Robot BEV v3 indexes to BEVFusion-compatible infos."""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_generation.robot_bev.schema import normalize_relative_path
from data_generation.robot_bev.validator import validate_dataset


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a canonical float32 [w, x, y, z] quaternion for a rotation."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    first_nonzero = next((value for value in quaternion if abs(value) > 1e-12), 0.0)
    if first_nonzero < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)


def _history_to_current(
    cur: Mapping[str, Any], hist: Mapping[str, Any]
) -> np.ndarray:
    cur_map_from_lidar = cur["T_map_base"] @ cur["lidar2base"]
    hist_map_from_lidar = hist["T_map_base"] @ hist["lidar2base"]
    return np.linalg.inv(cur_map_from_lidar) @ hist_map_from_lidar


def _make_sweeps(
    current: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    max_sweeps: int,
) -> List[Dict[str, Any]]:
    if max_sweeps <= 0:
        return []
    sweeps: List[Dict[str, Any]] = []
    for hist in list(history)[-max_sweeps:][::-1]:
        transform = _history_to_current(current, hist)
        sweeps.append(
            {
                "data_path": normalize_relative_path(hist["lidar_path"]),
                "timestamp": int(hist["timestamp"]),
                "sensor2lidar_rotation": transform[:3, :3].astype(np.float32),
                "sensor2lidar_translation": transform[:3, 3].astype(np.float32),
            }
        )
    return sweeps


def _relative_optional_path(raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    return None if value is None else normalize_relative_path(value)


def _convert_frame(
    raw: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    max_sweeps: int,
    camera_name: str,
) -> Dict[str, Any]:
    camera2base = np.asarray(raw["camera2base"], dtype=np.float32)
    lidar2base = np.asarray(raw["lidar2base"], dtype=np.float32)
    map_from_base = np.asarray(raw["T_map_base"], dtype=np.float32)
    camera2lidar = np.linalg.inv(lidar2base) @ camera2base
    camera = {
        "data_path": normalize_relative_path(raw["image_path"]),
        "sensor2ego_rotation": _rotation_to_quaternion(camera2base[:3, :3]),
        "sensor2ego_translation": camera2base[:3, 3].astype(np.float32),
        "ego2global_rotation": _rotation_to_quaternion(map_from_base[:3, :3]),
        "ego2global_translation": map_from_base[:3, 3].astype(np.float32),
        "sensor2lidar_rotation": camera2lidar[:3, :3].astype(np.float32),
        "sensor2lidar_translation": camera2lidar[:3, 3].astype(np.float32),
        "cam_intrinsic": np.asarray(raw["cam_intrinsic"], dtype=np.float32),
    }
    info = {
        "lidar_path": normalize_relative_path(raw["lidar_path"]),
        "token": raw["token"],
        "prev_token": raw["prev_token"],
        "timestamp": int(raw["timestamp"]),
        "sweeps": _make_sweeps(raw, history, max_sweeps),
        "cams": {camera_name: camera},
        "lidar2ego_translation": lidar2base[:3, 3].astype(np.float32),
        "lidar2ego_rotation": _rotation_to_quaternion(lidar2base[:3, :3]),
        "ego2global_translation": map_from_base[:3, 3].astype(np.float32),
        "ego2global_rotation": _rotation_to_quaternion(map_from_base[:3, :3]),
        "bev_mask_path": normalize_relative_path(raw["bev_mask_path"]),
        "bev_observed_mask_path": normalize_relative_path(
            raw["bev_observed_mask_path"]
        ),
        "bev_supervision_mask_path": _relative_optional_path(
            raw, "bev_supervision_mask_path"
        ),
        "class_validity": np.asarray(raw["class_validity"], dtype=np.uint8).copy(),
        "gt_boxes": np.empty((0, 7), dtype=np.float32),
        "gt_names": np.empty((0,), dtype="<U1"),
        "gt_velocity": np.empty((0, 2), dtype=np.float32),
        "num_lidar_pts": np.empty((0,), dtype=np.int64),
        "num_radar_pts": np.empty((0,), dtype=np.int64),
        "valid_flag": np.empty((0,), dtype=bool),
    }
    for key in ("depth_path", "semantic_path"):
        value = _relative_optional_path(raw, key)
        if value is not None:
            info[key] = value
    return info


def convert_split(
    root: Path, split: str, max_sweeps: int = 5, camera_name: str = "CAM_FRONT"
) -> Path:
    """Validate and convert one canonical index into a BEVFusion info pickle."""
    if max_sweeps < 0:
        raise ValueError("max_sweeps must be non-negative")
    root = Path(root).expanduser().resolve()
    validate_dataset(root, split)
    source = _load_pickle(root / f"robot_infos_{split}.pkl")
    history_by_scene: Dict[str, List[Mapping[str, Any]]] = {}
    converted: List[Dict[str, Any]] = []
    for raw in source["infos"]:
        scene_history = history_by_scene.setdefault(raw["scene_id"], [])
        converted.append(
            _convert_frame(raw, scene_history, max_sweeps, camera_name)
        )
        scene_history.append(raw)
    metadata = dict(source["metadata"])
    source_schema_version = int(metadata.get("schema_version", 4))
    metadata.update(
        {
            "version": f"robot-bev-v{source_schema_version}",
            "converter": "robot_bev_converter_v1",
            "source_schema_name": "robot_bev_dataset",
            "source_schema_version": source_schema_version,
        }
    )
    output = root / f"bevfusion_infos_{split}.pkl"
    _atomic_pickle(output, {"infos": converted, "metadata": metadata})
    return output


def _parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert canonical Robot BEV indexes for BEVFusion."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="all"
    )
    parser.add_argument("--max-sweeps", type=int, default=5)
    parser.add_argument("--camera-name", default="CAM_FRONT")
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = _parse_args(argv)
    if args.max_sweeps < 0:
        raise ValueError("--max-sweeps must be non-negative")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        output = convert_split(
            args.root, split, args.max_sweeps, args.camera_name
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
