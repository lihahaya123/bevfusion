import hashlib
import json
import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence

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
    effective_supervision_mask,
    normalize_relative_path,
)


@dataclass(frozen=True)
class FramePayload:
    frame_id: int
    timestamp: int
    rgb: np.ndarray
    points: np.ndarray
    bev_labels: np.ndarray
    observed_mask: np.ndarray
    class_validity: np.ndarray
    cam_intrinsic: np.ndarray
    camera2base: np.ndarray
    lidar2base: np.ndarray
    map_from_base: np.ndarray
    per_class_supervision_mask: Optional[np.ndarray] = None
    depth_mm: Optional[np.ndarray] = None
    semantics: Optional[np.ndarray] = None


class RobotBEVWriter:
    _SPLIT_NAMES = ("train", "val", "test")

    def __init__(
        self,
        root: Path,
        dataset_id: str,
        source_type: str,
        source_dataset: str,
        generator_name: str,
        generator_version: str,
        splits: Mapping[str, Sequence[str]],
        generation_parameters: Mapping[str, object],
        resume: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_id = dataset_id
        self._requested_split_names = set(splits)
        self.splits = {
            name: list(splits.get(name, [])) for name in self._SPLIT_NAMES
        }
        self.generation_parameters = dict(generation_parameters)
        self.metadata = self._build_metadata(
            source_type, source_dataset, generator_name, generator_version
        )
        self._validate_splits()
        self._initialize_root(resume)

    def write_frame(
        self, scene_id: str, split: str, frame: FramePayload
    ) -> Dict[str, object]:
        frame = self._validate_frame(scene_id, split, frame)
        record = self._write_frame_artifacts(scene_id, frame)
        self._append_manifest(scene_id, record)
        return record

    def finalize_scene(self, scene_id: str, split: str) -> Dict[str, object]:
        self._validate_scene_split(scene_id, split)
        records = self._load_manifest(scene_id)
        infos = [
            self._manifest_to_info(scene_id, index, record)
            for index, record in enumerate(records)
        ]
        payload = self._index_payload(infos, scene_split=split)
        self._atomic_pickle(self.root / scene_id / "scene_infos.pkl", payload)
        self._atomic_pickle(
            self.root / scene_id / f"robot_infos_{split}.pkl", payload
        )
        summary = self._scene_summary(scene_id, split, records)
        self._atomic_json(self.root / scene_id / "summary.json", summary)
        return summary

    def finalize_dataset(self) -> Dict[str, object]:
        split_infos = {name: [] for name in self._SPLIT_NAMES}
        summaries: List[Dict[str, object]] = []
        for split, scene_ids in self.splits.items():
            for scene_id in scene_ids:
                scene_payload = self._read_pickle(
                    self.root / scene_id / "scene_infos.pkl"
                )
                split_infos[split].extend(scene_payload["infos"])
                summaries.append(
                    self._read_json(self.root / scene_id / "summary.json")
                )
        for split, infos in split_infos.items():
            self._atomic_pickle(
                self.root / f"robot_infos_{split}.pkl",
                self._index_payload(infos),
            )
        root_summary = {
            "status": "complete",
            "info_counts": {
                name: len(infos) for name, infos in split_infos.items()
            },
            "scene_summaries": summaries,
            "failures": [],
        }
        self._atomic_json(self.root / "multi_scene_summary.json", root_summary)
        return root_summary

    def _build_metadata(
        self,
        source_type: str,
        source_dataset: str,
        generator_name: str,
        generator_version: str,
    ) -> Dict[str, object]:
        schema_contract = {
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
        fingerprint_payload = dict(schema_contract)
        fingerprint_payload["generation_parameters"] = self.generation_parameters
        serialized = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=self._json_default,
        ).encode("utf-8")
        metadata = dict(schema_contract)
        metadata.update(
            {
                "dataset_id": self.dataset_id,
                "source_type": source_type,
                "source_dataset": source_dataset,
                "generator": {
                    "name": generator_name,
                    "version": generator_version,
                },
                "generation_parameters": self.generation_parameters,
                "generation_fingerprint": hashlib.sha256(serialized).hexdigest(),
            }
        )
        return metadata

    def _validate_splits(self) -> None:
        unknown = self._requested_split_names.difference(self._SPLIT_NAMES)
        if unknown:
            raise SchemaError(f"unknown split names: {sorted(unknown)}")

        owner: Dict[str, str] = {}
        for split, scene_ids in self.splits.items():
            if len(scene_ids) != len(set(scene_ids)):
                raise SchemaError(f"duplicate scene IDs in split {split!r}")
            for scene_id in scene_ids:
                if not isinstance(scene_id, str) or not scene_id:
                    raise SchemaError("scene IDs must be non-empty strings")
                if normalize_relative_path(scene_id) != scene_id:
                    raise SchemaError(
                        f"scene ID must be a normalized POSIX path: {scene_id!r}"
                    )
                canonical_token(self.dataset_id, scene_id, 0)
                previous_split = owner.setdefault(scene_id, split)
                if previous_split != split:
                    raise SchemaError(
                        f"scene {scene_id!r} appears in both "
                        f"{previous_split!r} and {split!r}"
                    )

    def _initialize_root(self, resume: bool) -> None:
        metadata_path = self.root / "dataset_metadata.json"
        splits_path = self.root / "splits.json"

        if resume:
            if not metadata_path.is_file() or not splits_path.is_file():
                raise RuntimeError("cannot resume an uninitialized dataset root")
            existing_metadata = self._read_json(metadata_path)
            existing_splits = self._read_json(splits_path)
            if existing_metadata.get("generation_fingerprint") != self.metadata.get(
                "generation_fingerprint"
            ):
                raise RuntimeError("generation fingerprint mismatch")
            if existing_splits != self.splits:
                raise RuntimeError("dataset split mismatch")
            comparable_keys = (
                "schema_name",
                "schema_version",
                "dataset_id",
                "source_type",
                "source_dataset",
                "generator",
                "map_classes",
                "bev",
                "points",
            )
            if any(
                existing_metadata.get(key) != self.metadata.get(key)
                for key in comparable_keys
            ):
                raise RuntimeError("dataset metadata mismatch")
            return

        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise RuntimeError(
                "dataset root is not empty; use a new root or resume=True"
            )
        self._atomic_json(metadata_path, self.metadata)
        self._atomic_json(splits_path, self.splits)

    def _validate_scene_split(self, scene_id: str, split: str) -> None:
        if split not in self._SPLIT_NAMES:
            raise SchemaError(f"unknown split name: {split!r}")
        if scene_id not in self.splits[split]:
            raise SchemaError(
                f"scene {scene_id!r} does not belong to split {split!r}"
            )

    def _validate_frame(
        self, scene_id: str, split: str, frame: FramePayload
    ) -> FramePayload:
        self._validate_scene_split(scene_id, split)
        records = self._load_manifest(scene_id)
        expected_frame_id = len(records)
        if not isinstance(frame.frame_id, (int, np.integer)) or isinstance(
            frame.frame_id, (bool, np.bool_)
        ):
            raise SchemaError("frame_id must be an integer")
        if int(frame.frame_id) != expected_frame_id:
            raise SchemaError(
                f"frame_id {frame.frame_id} is not contiguous; "
                f"expected {expected_frame_id}"
            )
        if not isinstance(frame.timestamp, (int, np.integer)) or isinstance(
            frame.timestamp, (bool, np.bool_)
        ):
            raise SchemaError("timestamp must be an integer number of microseconds")
        if records and int(frame.timestamp) <= int(records[-1]["timestamp"]):
            raise SchemaError("timestamps must be strictly increasing within a scene")

        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise SchemaError("rgb must have shape [H,W,3] and dtype uint8")

        points = np.asarray(frame.points)
        if points.ndim != 2 or points.shape[1] != len(POINT_DIMENSIONS):
            raise SchemaError("points must have shape [N,5]")
        points = self._finite_float32(points, "points")

        labels = self._binary_uint8(frame.bev_labels, BEV_SHAPE, "bev_labels")
        observed = np.asarray(frame.observed_mask)
        class_validity = np.asarray(frame.class_validity)
        supervision_mask = (
            None
            if frame.per_class_supervision_mask is None
            else np.asarray(frame.per_class_supervision_mask)
        )
        effective_supervision_mask(
            observed,
            class_validity,
            supervision_mask,
        )
        if np.any(labels[:, observed == 0]):
            raise SchemaError("BEV labels outside the observed mask must be zero")

        matrices = (
            ("cam_intrinsic", frame.cam_intrinsic, (3, 3)),
            ("camera2base", frame.camera2base, (4, 4)),
            ("lidar2base", frame.lidar2base, (4, 4)),
            ("map_from_base", frame.map_from_base, (4, 4)),
        )
        converted_matrices: Dict[str, np.ndarray] = {}
        for name, value, shape in matrices:
            matrix = np.asarray(value)
            if matrix.shape != shape:
                raise SchemaError(f"{name} shape {matrix.shape} != {shape}")
            converted_matrices[name] = self._finite_float32(matrix, name)

        image_shape = rgb.shape[:2]
        optional_images: Dict[str, Optional[np.ndarray]] = {
            "depth_mm": None,
            "semantics": None,
        }
        for name, value in (
            ("depth_mm", frame.depth_mm),
            ("semantics", frame.semantics),
        ):
            if value is None:
                continue
            image = np.asarray(value)
            if (
                image.ndim != 2
                or image.shape != image_shape
                or image.dtype != np.uint16
            ):
                raise SchemaError(
                    f"{name} must be a two-dimensional uint16 array "
                    "matching RGB height/width"
                )
            optional_images[name] = image

        return replace(
            frame,
            frame_id=int(frame.frame_id),
            timestamp=int(frame.timestamp),
            rgb=rgb,
            points=points,
            bev_labels=labels,
            observed_mask=observed,
            class_validity=class_validity,
            cam_intrinsic=converted_matrices["cam_intrinsic"],
            camera2base=converted_matrices["camera2base"],
            lidar2base=converted_matrices["lidar2base"],
            map_from_base=converted_matrices["map_from_base"],
            per_class_supervision_mask=supervision_mask,
            depth_mm=optional_images["depth_mm"],
            semantics=optional_images["semantics"],
        )

    @staticmethod
    def _finite_float32(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise SchemaError(
                f"{name} must contain real values representable as float32"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(array, dtype=np.float32)
        if not np.isfinite(converted).all():
            raise SchemaError(
                f"{name} must remain finite when converted to float32"
            )
        return np.ascontiguousarray(converted)

    @staticmethod
    def _binary_uint8(
        value: np.ndarray, shape: Sequence[int], name: str
    ) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != tuple(shape):
            raise SchemaError(f"{name} shape {array.shape} != {tuple(shape)}")
        if array.dtype != np.uint8:
            raise SchemaError(f"{name} dtype {array.dtype} != uint8")
        if not np.isin(array, (0, 1)).all():
            raise SchemaError(f"{name} must be binary")
        return array

    def _write_frame_artifacts(
        self, scene_id: str, frame: FramePayload
    ) -> Dict[str, object]:
        stem = f"{int(frame.frame_id):06d}"
        paths = {
            "image_path": self._relative_path(scene_id, "images", stem + ".png"),
            "lidar_path": self._relative_path(scene_id, "points", stem + ".bin"),
            "bev_mask_path": self._relative_path(
                scene_id, "bev_masks", stem + ".npy"
            ),
            "bev_observed_mask_path": self._relative_path(
                scene_id, "bev_observed_masks", stem + ".npy"
            ),
            "bev_supervision_mask_path": None,
            "depth_path": None,
            "semantic_path": None,
        }
        if frame.per_class_supervision_mask is not None:
            paths["bev_supervision_mask_path"] = self._relative_path(
                scene_id, "bev_supervision_masks", stem + ".npy"
            )
        if frame.depth_mm is not None:
            paths["depth_path"] = self._relative_path(
                scene_id, "depths", stem + ".png"
            )
        if frame.semantics is not None:
            paths["semantic_path"] = self._relative_path(
                scene_id, "semantics", stem + ".png"
            )

        self._atomic_png(self.root / str(paths["image_path"]), frame.rgb)
        self._atomic_points(self.root / str(paths["lidar_path"]), frame.points)
        self._atomic_npy(
            self.root / str(paths["bev_mask_path"]), frame.bev_labels
        )
        self._atomic_npy(
            self.root / str(paths["bev_observed_mask_path"]),
            frame.observed_mask,
        )
        if paths["bev_supervision_mask_path"] is not None:
            self._atomic_npy(
                self.root / str(paths["bev_supervision_mask_path"]),
                frame.per_class_supervision_mask,
            )
        if paths["depth_path"] is not None:
            self._atomic_png(
                self.root / str(paths["depth_path"]), frame.depth_mm
            )
        if paths["semantic_path"] is not None:
            self._atomic_png(
                self.root / str(paths["semantic_path"]), frame.semantics
            )

        labels = np.asarray(frame.bev_labels)
        record: Dict[str, object] = {
            "dataset_id": self.dataset_id,
            "scene_id": scene_id,
            "frame_id": int(frame.frame_id),
            "timestamp": int(frame.timestamp),
            **paths,
            "class_validity": np.asarray(frame.class_validity).tolist(),
            "cam_intrinsic": self._finite_float32(
                frame.cam_intrinsic, "cam_intrinsic"
            ).tolist(),
            "camera2base": self._finite_float32(
                frame.camera2base, "camera2base"
            ).tolist(),
            "lidar2base": self._finite_float32(
                frame.lidar2base, "lidar2base"
            ).tolist(),
            "T_map_base": self._finite_float32(
                frame.map_from_base, "map_from_base"
            ).tolist(),
            "pose_valid": True,
            "point_count": int(np.asarray(frame.points).shape[0]),
            "per_class_sums": [
                int(value) for value in labels.sum(axis=(1, 2), dtype=np.int64)
            ],
            "observed_sum": int(
                np.asarray(frame.observed_mask).sum(dtype=np.int64)
            ),
        }
        return record

    def _append_manifest(
        self, scene_id: str, record: Mapping[str, object]
    ) -> None:
        manifest_path = self.root / scene_id / "manifest.jsonl"
        records = self._load_manifest(scene_id)
        records.append(dict(record))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path(manifest_path)
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                for item in records:
                    json.dump(
                        item,
                        handle,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        default=self._json_default,
                    )
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(manifest_path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _load_manifest(self, scene_id: str) -> List[Dict[str, object]]:
        manifest_path = self.root / scene_id / "manifest.jsonl"
        if not manifest_path.exists():
            return []
        records: List[Dict[str, object]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RuntimeError(
                        f"blank line in manifest {manifest_path} at {line_number}"
                    )
                records.append(json.loads(line))
        return records

    def _manifest_to_info(
        self,
        scene_id: str,
        index: int,
        record: Mapping[str, object],
    ) -> Dict[str, object]:
        frame_id = int(record["frame_id"])
        info: Dict[str, object] = {
            "dataset_id": self.dataset_id,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "token": canonical_token(self.dataset_id, scene_id, frame_id),
            "prev_token": (
                ""
                if index == 0
                else canonical_token(self.dataset_id, scene_id, frame_id - 1)
            ),
            "timestamp": int(record["timestamp"]),
            "image_path": normalize_relative_path(str(record["image_path"])),
            "lidar_path": normalize_relative_path(str(record["lidar_path"])),
            "bev_mask_path": normalize_relative_path(str(record["bev_mask_path"])),
            "bev_observed_mask_path": normalize_relative_path(
                str(record["bev_observed_mask_path"])
            ),
            "bev_supervision_mask_path": record.get(
                "bev_supervision_mask_path"
            ),
            "class_validity": np.asarray(
                record["class_validity"], dtype=np.uint8
            ),
            "cam_intrinsic": np.asarray(
                record["cam_intrinsic"], dtype=np.float32
            ),
            "camera2base": np.asarray(record["camera2base"], dtype=np.float32),
            "lidar2base": np.asarray(record["lidar2base"], dtype=np.float32),
            "T_map_base": np.asarray(record["T_map_base"], dtype=np.float32),
            "pose_valid": bool(record["pose_valid"]),
            "depth_path": record.get("depth_path"),
            "semantic_path": record.get("semantic_path"),
        }
        for key in (
            "bev_supervision_mask_path",
            "depth_path",
            "semantic_path",
        ):
            if info[key] is not None:
                info[key] = normalize_relative_path(str(info[key]))
        return info

    def _scene_summary(
        self,
        scene_id: str,
        split: str,
        records: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        point_counts = [int(record["point_count"]) for record in records]
        per_class_sums = np.zeros((len(MAP_CLASSES),), dtype=np.int64)
        for record in records:
            per_class_sums += np.asarray(record["per_class_sums"], dtype=np.int64)
        return {
            "status": "complete",
            "scene_id": scene_id,
            "split": split,
            "generation_fingerprint": self.metadata["generation_fingerprint"],
            "frame_count": len(records),
            "point_count": {
                "min": min(point_counts) if point_counts else 0,
                "max": max(point_counts) if point_counts else 0,
                "mean": float(np.mean(point_counts)) if point_counts else 0.0,
            },
            "per_class_sums": {
                name: int(value)
                for name, value in zip(MAP_CLASSES, per_class_sums.tolist())
            },
            "observed_sum": sum(
                int(record["observed_sum"]) for record in records
            ),
        }

    def _index_payload(
        self,
        infos: Sequence[Mapping[str, object]],
        scene_split: Optional[str] = None,
    ) -> Dict[str, object]:
        metadata = dict(self.metadata)
        if scene_split is not None:
            metadata["scene_split"] = scene_split
        return {"metadata": metadata, "infos": list(infos)}

    @staticmethod
    def _relative_path(scene_id: str, directory: str, filename: str) -> str:
        return normalize_relative_path(
            PurePosixPath(scene_id) / directory / filename
        )

    @staticmethod
    def _temp_path(path: Path) -> Path:
        return path.with_name(path.name + ".tmp")

    def _atomic_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path(path)
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=self._json_default,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _atomic_pickle(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path(path)
        try:
            with temp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _atomic_npy(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path(path)
        try:
            with temp_path.open("wb") as handle:
                np.save(handle, np.asarray(array), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _atomic_png(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path(path)
        try:
            values = np.asarray(array)
            if values.ndim == 2 and values.dtype == np.uint16:
                image = Image.fromarray(values, mode="I;16")
            elif (
                values.ndim == 3
                and values.shape[2] == 3
                and values.dtype == np.uint8
            ):
                image = Image.fromarray(values, mode="RGB")
            else:
                raise SchemaError(
                    "PNG arrays must be uint8 RGB or uint16 grayscale"
                )
            with temp_path.open("wb") as handle:
                image.save(handle, format="PNG")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _atomic_points(self, path: Path, points: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temp_path(path)
        try:
            encoded = self._finite_float32(points, "points").tobytes()
            with temp_path.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_json(path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_pickle(path: Path) -> Dict[str, object]:
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
