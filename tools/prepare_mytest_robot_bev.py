"""Prepare self-collected mytest data as canonical RobotBEV v4 data.

The current mytest source only contains RGB images, camera intrinsics and
camera/depth point clouds.  This script therefore produces an inference-only
RobotBEV dataset by default: BEV labels and observed masks are written as
empty placeholders and all classes are marked invalid.  That keeps the format
compatible with the RobotBEV/BEVFusion pipeline without pretending that ground
truth labels exist.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.robot_bev.schema import (  # noqa: E402
    BEV_SHAPE,
    BEV_XBOUND,
    BEV_YBOUND,
    BEV_ZBOUND,
    MAP_CLASSES,
    OBSERVED_MASK_SHAPE,
    POINT_DIMENSIONS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    canonical_token,
    normalize_relative_path,
)
from data_generation.robot_bev.validator import validate_dataset  # noqa: E402
from tools.data_converter.robot_bev_converter import convert_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert data/mytest self-collected frames to RobotBEV v4."
    )
    parser.add_argument(
        "--src-root",
        default="data/mytest/data",
        type=Path,
        help="Input directory containing rgb/, pclCam/ and in.txt.",
    )
    parser.add_argument(
        "--out-root",
        default="data/mytest/robot_bev",
        type=Path,
        help="Output RobotBEV dataset root.",
    )
    parser.add_argument("--dataset-id", default="mytest_robot_bev_v4")
    parser.add_argument("--scene-id", default="mytest")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Split to place all converted mytest frames into.",
    )
    parser.add_argument(
        "--camera2base",
        default=None,
        type=Path,
        help=(
            "Optional 4x4 or 3x4 camera-to-robot-base matrix. Column-vector "
            "convention: p_base = R @ p_camera + t. If omitted, identity is used."
        ),
    )
    parser.add_argument(
        "--point-scale",
        type=float,
        default=0.001,
        help="Scale applied to txt point coordinates before transforms.",
    )
    parser.add_argument(
        "--points-coord",
        choices=("camera", "base"),
        default="camera",
        help=(
            "Coordinate frame of pclCam txt points. Use 'camera' to transform "
            "by --camera2base; use 'base' when points are already in robot base."
        ),
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help=(
            "History sweeps to write into BEVFusion infos. Default 0 because "
            "mytest has no reliable ego poses yet."
        ),
    )
    parser.add_argument("--camera-name", default="CAM_FRONT")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --out-root before writing.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only write RobotBEV raw indexes, do not create bevfusion_infos_*.pkl.",
    )
    return parser.parse_args()


def load_intrinsic(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]
    if len(values) < 4:
        raise ValueError(f"Expected fx, fy, cx, cy in {path}, got: {text!r}")
    fx, fy, cx, cy = values[:4]
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def load_transform(path: Optional[Path]) -> np.ndarray:
    if path is None:
        print("WARNING: --camera2base not provided; using identity transform.")
        return np.eye(4, dtype=np.float32)
    mat = np.loadtxt(path, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    raise ValueError(f"{path} must contain a 4x4 or 3x4 matrix, got {mat.shape}")


def frame_id_from_rgb(path: Path) -> int:
    return int(path.stem)


def frame_id_from_pointcloud(path: Path) -> Optional[int]:
    match = re.search(r"_LOS_(\d+)_", path.name)
    return None if match is None else int(match.group(1))


def collect_point_files(pcl_dir: Path) -> Dict[int, Path]:
    files: Dict[int, Path] = {}
    duplicates: Dict[int, List[Path]] = {}
    for path in sorted(pcl_dir.glob("*.txt")):
        raw_id = frame_id_from_pointcloud(path)
        if raw_id is None:
            continue
        if raw_id in files:
            duplicates.setdefault(raw_id, [files[raw_id]]).append(path)
            continue
        files[raw_id] = path
    for raw_id, paths in sorted(duplicates.items()):
        names = ", ".join(p.name for p in paths)
        print(f"WARNING: duplicate pclCam files for raw frame {raw_id}; using {paths[0].name}; candidates: {names}")
    return files


def read_points_txt(path: Path, scale: float) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"{path} must contain at least xyz columns")
    return points[:, :3].astype(np.float32) * np.float32(scale)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return points @ rot.T + trans


def rel(path: Path, root: Path) -> str:
    return normalize_relative_path(path.relative_to(root).as_posix())


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(to_jsonable(record), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_metadata(args: argparse.Namespace, fingerprint: str) -> Dict[str, object]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "map_classes": list(MAP_CLASSES),
        "bev": {
            "xbound": list(BEV_XBOUND),
            "ybound": list(BEV_YBOUND),
            "zbound": list(BEV_ZBOUND),
            "shape": list(BEV_SHAPE),
            "encoding": "uint8_multihot",
            "observed_mask_shape": list(OBSERVED_MASK_SHAPE),
        },
        "points": {
            "dtype": "float32",
            "dimensions": list(POINT_DIMENSIONS),
        },
        "dataset_id": args.dataset_id,
        "source_type": "real",
        "source_dataset": "mytest",
        "generator": {
            "name": "prepare_mytest_robot_bev",
            "version": "1",
        },
        "generation_parameters": {
            "src_root": str(args.src_root),
            "dataset_id": args.dataset_id,
            "scene_id": args.scene_id,
            "split": args.split,
            "point_scale": args.point_scale,
            "points_coord": args.points_coord,
            "camera2base": None if args.camera2base is None else str(args.camera2base),
            "max_sweeps": args.max_sweeps,
            "inference_only": True,
            "label_policy": "empty_placeholder_masks_class_validity_zero",
        },
        "generation_fingerprint": fingerprint,
    }


def fingerprint_inputs(args: argparse.Namespace, rgb_files: Sequence[Path]) -> str:
    digest = hashlib.sha1()
    digest.update(str(args.src_root).encode("utf-8"))
    digest.update(str(args.dataset_id).encode("utf-8"))
    digest.update(str(args.scene_id).encode("utf-8"))
    for path in rgb_files:
        digest.update(path.name.encode("utf-8"))
    return digest.hexdigest()


def build_infos(args: argparse.Namespace) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    src_root = args.src_root
    out_root = args.out_root
    scene_dir = out_root / args.scene_id
    image_dir = scene_dir / "images"
    points_dir = scene_dir / "points"
    mask_dir = scene_dir / "bev_masks"
    observed_dir = scene_dir / "bev_observed_masks"
    for directory in (image_dir, points_dir, mask_dir, observed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    intrinsic = load_intrinsic(src_root / "in.txt")
    camera2base = load_transform(args.camera2base)
    lidar2base = np.eye(4, dtype=np.float32)
    map_from_base = np.eye(4, dtype=np.float32)
    pcl_files = collect_point_files(src_root / "pclCam")
    rgb_files = sorted((src_root / "rgb").glob("*.png"), key=frame_id_from_rgb)
    metadata = make_metadata(args, fingerprint_inputs(args, rgb_files))

    empty_bev = np.zeros(BEV_SHAPE, dtype=np.uint8)
    empty_observed = np.zeros(OBSERVED_MASK_SHAPE, dtype=np.uint8)
    class_validity = np.zeros((len(MAP_CLASSES),), dtype=np.uint8)
    infos: List[Dict[str, object]] = []
    manifest_records: List[Dict[str, object]] = []
    prev_token = ""

    for frame_id, rgb_path in enumerate(rgb_files):
        raw_frame_id = frame_id_from_rgb(rgb_path)
        pcl_path = pcl_files.get(raw_frame_id)
        if pcl_path is None:
            print(f"WARNING: skip raw frame {raw_frame_id}, no matching pclCam txt.")
            continue

        token = canonical_token(args.dataset_id, args.scene_id, len(infos))
        output_stem = f"{len(infos):06d}"
        image_path = image_dir / f"{output_stem}.png"
        point_path = points_dir / f"{output_stem}.bin"
        mask_path = mask_dir / f"{output_stem}.npy"
        observed_path = observed_dir / f"{output_stem}.npy"

        with Image.open(rgb_path) as image:
            image.convert("RGB").save(image_path)

        points = read_points_txt(pcl_path, args.point_scale)
        if args.points_coord == "camera":
            points = transform_points(points, camera2base)
        attrs = np.zeros((points.shape[0], 2), dtype=np.float32)
        points_5d = np.concatenate([points.astype(np.float32), attrs], axis=1)
        points_5d.tofile(point_path)

        np.save(mask_path, empty_bev)
        np.save(observed_path, empty_observed)

        info = {
            "dataset_id": args.dataset_id,
            "scene_id": args.scene_id,
            "frame_id": len(infos),
            "raw_frame_id": raw_frame_id,
            "token": token,
            "prev_token": prev_token,
            "timestamp": int(raw_frame_id),
            "image_path": rel(image_path, out_root),
            "lidar_path": rel(point_path, out_root),
            "bev_mask_path": rel(mask_path, out_root),
            "bev_observed_mask_path": rel(observed_path, out_root),
            "bev_supervision_mask_path": None,
            "class_validity": class_validity.copy(),
            "cam_intrinsic": intrinsic.copy(),
            "camera2base": camera2base.copy(),
            "lidar2base": lidar2base.copy(),
            "T_map_base": map_from_base.copy(),
            "pose_valid": True,
        }
        infos.append(info)
        manifest_record = dict(info)
        manifest_record.update(
            {
                "point_count": int(points_5d.shape[0]),
                "per_class_sums": [0] * len(MAP_CLASSES),
                "observed_sum": 0,
            }
        )
        manifest_records.append(manifest_record)
        prev_token = token

    if not infos:
        raise RuntimeError(f"No frames were converted from {src_root}")

    write_jsonl(scene_dir / "manifest.jsonl", manifest_records)
    summary = {
        "status": "complete",
        "scene_id": args.scene_id,
        "split": args.split,
        "generation_fingerprint": metadata["generation_fingerprint"],
        "frame_count": len(infos),
        "inference_only": True,
    }
    write_json(scene_dir / "summary.json", summary)
    scene_payload = {
        "metadata": {**metadata, "scene_split": args.split},
        "infos": infos,
    }
    write_pickle(scene_dir / "scene_infos.pkl", scene_payload)
    write_pickle(scene_dir / f"robot_infos_{args.split}.pkl", scene_payload)
    return metadata, infos


def write_root_indexes(
    out_root: Path,
    metadata: Mapping[str, object],
    scene_id: str,
    split: str,
    infos: Sequence[Mapping[str, object]],
    scene_summary: Mapping[str, object],
) -> None:
    write_json(out_root / "dataset_metadata.json", metadata)
    splits = {"train": [], "val": [], "test": []}
    splits[split] = [scene_id]
    write_json(out_root / "splits.json", splits)
    counts = {"train": 0, "val": 0, "test": 0}
    counts[split] = len(infos)
    write_json(
        out_root / "multi_scene_summary.json",
        {
            "status": "complete",
            "dataset_id": metadata["dataset_id"],
            "generation_fingerprint": metadata["generation_fingerprint"],
            "info_counts": counts,
            "splits": splits,
            "scene_summaries": [dict(scene_summary)],
            "inference_only": True,
        },
    )
    for split_name in ("train", "val", "test"):
        split_infos = list(infos) if split_name == split else []
        write_pickle(
            out_root / f"robot_infos_{split_name}.pkl",
            {"metadata": dict(metadata), "infos": split_infos},
        )


def main() -> int:
    args = parse_args()
    args.src_root = args.src_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if args.max_sweeps < 0:
        raise ValueError("--max-sweeps must be non-negative")
    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    metadata, infos = build_infos(args)
    scene_summary = {
        "status": "complete",
        "scene_id": args.scene_id,
        "split": args.split,
        "generation_fingerprint": metadata["generation_fingerprint"],
        "frame_count": len(infos),
        "inference_only": True,
    }
    write_root_indexes(
        args.out_root, metadata, args.scene_id, args.split, infos, scene_summary
    )

    report = validate_dataset(args.out_root)
    print(f"Validated RobotBEV dataset: {report.frame_counts}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")

    if not args.skip_convert:
        for split in ("train", "val", "test"):
            output = convert_split(
                args.out_root,
                split,
                max_sweeps=args.max_sweeps,
                camera_name=args.camera_name,
            )
            print(f"Wrote {output}")

    print(f"Converted {len(infos)} frames to {args.out_root}")
    print("NOTE: generated BEV labels are empty inference-only placeholders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
