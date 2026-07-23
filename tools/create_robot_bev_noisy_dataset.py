#!/usr/bin/env python3
"""Create a RobotBEV dataset copy with noisy point clouds for robustness tests.

The script keeps the dataset layout and info files unchanged. It links or copies
all non-point-cloud files into a new root, then rewrites only the selected split
point files with synthetic sensor noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


SPLITS = ("train", "val", "test")
PROFILES: Dict[str, Dict[str, float]] = {
    "light": {
        "radial_std_base": 0.005,
        "radial_std_per_m": 0.001,
        "lateral_std_base": 0.002,
        "lateral_std_per_m": 0.0005,
        "dropout_prob": 0.03,
        "dropout_per_m": 0.01,
        "max_dropout_prob": 0.15,
        "intensity_dropout_weight": 0.03,
        "intensity_std": 0.03,
        "outlier_ratio": 0.005,
        "sector_dropout_prob": 0.10,
        "sector_count": 1,
        "sector_width_deg": 5.0,
    },
    "mid": {
        "radial_std_base": 0.015,
        "radial_std_per_m": 0.003,
        "lateral_std_base": 0.005,
        "lateral_std_per_m": 0.001,
        "dropout_prob": 0.08,
        "dropout_per_m": 0.025,
        "max_dropout_prob": 0.35,
        "intensity_dropout_weight": 0.08,
        "intensity_std": 0.07,
        "outlier_ratio": 0.02,
        "sector_dropout_prob": 0.25,
        "sector_count": 1,
        "sector_width_deg": 8.0,
    },
    "heavy": {
        "radial_std_base": 0.030,
        "radial_std_per_m": 0.006,
        "lateral_std_base": 0.010,
        "lateral_std_per_m": 0.002,
        "dropout_prob": 0.15,
        "dropout_per_m": 0.05,
        "max_dropout_prob": 0.60,
        "intensity_dropout_weight": 0.15,
        "intensity_std": 0.12,
        "outlier_ratio": 0.05,
        "sector_dropout_prob": 0.40,
        "sector_count": 2,
        "sector_width_deg": 12.0,
    },
}


def require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "numpy is required to write noisy point clouds. "
            "Install numpy or run this script inside the BEVFusion environment."
        )


@dataclass(frozen=True)
class NoiseConfig:
    radial_std_base: float
    radial_std_per_m: float
    lateral_std_base: float
    lateral_std_per_m: float
    dropout_prob: float
    dropout_per_m: float
    max_dropout_prob: float
    intensity_dropout_weight: float
    intensity_std: float
    outlier_ratio: float
    sector_dropout_prob: float
    sector_count: int
    sector_width_deg: float
    min_points: int
    point_cloud_range: Tuple[float, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a separate RobotBEV dataset root whose selected split point "
            "clouds contain realistic measurement noise."
        )
    )
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=(*SPLITS, "all"),
        help="Dataset splits whose point files will be noised. Default: test.",
    )
    parser.add_argument(
        "--profile",
        choices=(*PROFILES.keys(), "custom"),
        default="mid",
        help="Noise preset. Use custom to rely entirely on explicit arguments.",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--load-dim", type=int, default=5)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize unchanged files in dst-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without creating files.",
    )
    parser.add_argument("--radial-std-base", type=float)
    parser.add_argument("--radial-std-per-m", type=float)
    parser.add_argument("--lateral-std-base", type=float)
    parser.add_argument("--lateral-std-per-m", type=float)
    parser.add_argument("--dropout-prob", type=float)
    parser.add_argument("--dropout-per-m", type=float)
    parser.add_argument("--max-dropout-prob", type=float)
    parser.add_argument("--intensity-dropout-weight", type=float)
    parser.add_argument("--intensity-std", type=float)
    parser.add_argument("--outlier-ratio", type=float)
    parser.add_argument("--sector-dropout-prob", type=float)
    parser.add_argument("--sector-count", type=int)
    parser.add_argument("--sector-width-deg", type=float)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=(0.0, -1.52, -0.5, 3.04, 1.52, 2.0),
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Range used when sampling outlier points.",
    )
    return parser.parse_args()


def selected_splits(raw_splits: Sequence[str]) -> Tuple[str, ...]:
    if "all" in raw_splits:
        return SPLITS
    unique = []
    for split in raw_splits:
        if split not in unique:
            unique.append(split)
    return tuple(unique)


def build_noise_config(args: argparse.Namespace) -> NoiseConfig:
    values = dict(PROFILES.get(args.profile, {}))
    for name in (
        "radial_std_base",
        "radial_std_per_m",
        "lateral_std_base",
        "lateral_std_per_m",
        "dropout_prob",
        "dropout_per_m",
        "max_dropout_prob",
        "intensity_dropout_weight",
        "intensity_std",
        "outlier_ratio",
        "sector_dropout_prob",
        "sector_count",
        "sector_width_deg",
    ):
        override = getattr(args, name)
        if override is not None:
            values[name] = override
    missing = sorted(set(PROFILES["mid"]) - set(values))
    if missing:
        raise ValueError(f"missing custom noise parameters: {missing}")
    return NoiseConfig(
        radial_std_base=float(values["radial_std_base"]),
        radial_std_per_m=float(values["radial_std_per_m"]),
        lateral_std_base=float(values["lateral_std_base"]),
        lateral_std_per_m=float(values["lateral_std_per_m"]),
        dropout_prob=float(values["dropout_prob"]),
        dropout_per_m=float(values["dropout_per_m"]),
        max_dropout_prob=float(values["max_dropout_prob"]),
        intensity_dropout_weight=float(values["intensity_dropout_weight"]),
        intensity_std=float(values["intensity_std"]),
        outlier_ratio=float(values["outlier_ratio"]),
        sector_dropout_prob=float(values["sector_dropout_prob"]),
        sector_count=int(values["sector_count"]),
        sector_width_deg=float(values["sector_width_deg"]),
        min_points=int(args.min_points),
        point_cloud_range=tuple(float(v) for v in args.point_cloud_range),
    )


def normalize_relative_path(value: str) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe dataset-relative path: {value!r}")
    return path.as_posix()


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def collect_point_paths_from_manifests(root: Path, splits: Sequence[str]) -> Set[str]:
    splits_path = root / "splits.json"
    if not splits_path.is_file():
        return set()
    with splits_path.open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)

    point_paths: Set[str] = set()
    for split in splits:
        scene_ids = split_payload.get(split)
        if not isinstance(scene_ids, list):
            raise ValueError(f"splits.json does not contain list split {split!r}")
        for scene_id in scene_ids:
            manifest_path = root / normalize_relative_path(scene_id) / "manifest.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"missing manifest: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    try:
                        point_paths.add(normalize_relative_path(record["lidar_path"]))
                    except KeyError as error:
                        raise ValueError(
                            f"{manifest_path}:{line_number} missing lidar_path"
                        ) from error
    return point_paths


def collect_point_paths_from_infos(root: Path, splits: Sequence[str]) -> Set[str]:
    point_paths: Set[str] = set()
    for split in splits:
        info_path = root / f"bevfusion_infos_{split}.pkl"
        if not info_path.is_file():
            info_path = root / f"robot_infos_{split}.pkl"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"cannot find bevfusion_infos_{split}.pkl or robot_infos_{split}.pkl"
            )
        payload = load_pickle(info_path)
        infos = payload.get("infos") if isinstance(payload, dict) else None
        if not isinstance(infos, list):
            raise ValueError(f"{info_path} does not contain an infos list")
        for info in infos:
            point_paths.add(normalize_relative_path(info["lidar_path"]))
            for sweep in info.get("sweeps", []):
                point_paths.add(normalize_relative_path(sweep["data_path"]))
    return point_paths


def collect_point_paths(root: Path, splits: Sequence[str]) -> Set[str]:
    point_paths = collect_point_paths_from_manifests(root, splits)
    if point_paths:
        return point_paths
    return collect_point_paths_from_infos(root, splits)


def iter_files(root: Path) -> Iterable[Tuple[Path, str]]:
    for current, _, filenames in os.walk(root):
        current_path = Path(current)
        for filename in filenames:
            source = current_path / filename
            yield source, source.relative_to(root).as_posix()


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def clone_dataset_shell(
    src_root: Path,
    dst_root: Path,
    noisy_point_paths: Set[str],
    copy_mode: str,
) -> int:
    linked = 0
    for source, rel_path in iter_files(src_root):
        destination = dst_root / rel_path
        if rel_path in noisy_point_paths:
            destination.parent.mkdir(parents=True, exist_ok=True)
            continue
        materialize_file(source, destination, copy_mode)
        linked += 1
    return linked


def load_points(path: Path, load_dim: int) -> np.ndarray:
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    else:
        points = np.fromfile(path, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, load_dim)
    if points.size % load_dim != 0:
        raise ValueError(f"{path} has {points.size} values, not divisible by {load_dim}")
    return points.reshape(-1, load_dim)


def save_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if path.suffix == ".npy":
        np.save(path, points)
    else:
        points.tofile(path)


def random_tangent_directions(
    unit: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    random_vec = rng.normal(size=unit.shape).astype(np.float32)
    projected = (random_vec * unit).sum(axis=1, keepdims=True) * unit
    tangent = random_vec - projected
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    fallback = norm[:, 0] < 1e-6
    if np.any(fallback):
        tangent[fallback] = np.stack(
            [-unit[fallback, 1], unit[fallback, 0], np.zeros(fallback.sum())],
            axis=1,
        )
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.maximum(norm, 1e-6)


def apply_sector_dropout(
    points: np.ndarray,
    keep: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.sector_count <= 0 or rng.random() >= config.sector_dropout_prob:
        return keep
    angles = np.arctan2(points[:, 1], points[:, 0])
    width = math.radians(config.sector_width_deg)
    for _ in range(config.sector_count):
        center = rng.uniform(-math.pi, math.pi)
        wrapped = np.abs((angles - center + math.pi) % (2.0 * math.pi) - math.pi)
        keep &= wrapped > width / 2.0
    return keep


def add_outliers(
    points: np.ndarray,
    count: int,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return points
    x_min, y_min, z_min, x_max, y_max, z_max = config.point_cloud_range
    outliers = np.zeros((count, points.shape[1]), dtype=np.float32)
    outliers[:, 0] = rng.uniform(x_min, x_max, count)
    outliers[:, 1] = rng.uniform(y_min, y_max, count)
    outliers[:, 2] = rng.uniform(z_min, z_max, count)
    if points.shape[1] > 3:
        outliers[:, 3] = rng.uniform(0.0, 0.25, count)
    if points.shape[1] > 4:
        if len(points):
            outliers[:, 4] = np.median(points[:, 4])
        else:
            outliers[:, 4] = 0.0
    return np.concatenate([points, outliers], axis=0)


def add_realistic_noise(
    points: np.ndarray, config: NoiseConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, Dict[str, float]]:
    if len(points) == 0:
        return points.astype(np.float32), {
            "input_points": 0,
            "kept_points": 0,
            "outlier_points": 0,
            "output_points": 0,
        }

    noisy = points.copy().astype(np.float32)
    xyz = noisy[:, :3]
    distance = np.linalg.norm(xyz, axis=1)
    unit = xyz / np.maximum(distance[:, None], 1e-6)

    radial_std = config.radial_std_base + config.radial_std_per_m * distance
    radial_noise = rng.normal(0.0, radial_std).astype(np.float32)
    xyz += unit * radial_noise[:, None]

    tangent = random_tangent_directions(unit, rng)
    lateral_std = config.lateral_std_base + config.lateral_std_per_m * distance
    lateral_noise = rng.normal(0.0, lateral_std).astype(np.float32)
    xyz += tangent * lateral_noise[:, None]

    dropout = config.dropout_prob + config.dropout_per_m * distance
    if noisy.shape[1] > 3:
        intensity = np.clip(noisy[:, 3], 0.0, 1.0)
        dropout += config.intensity_dropout_weight * (1.0 - intensity)
    dropout = np.clip(dropout, 0.0, config.max_dropout_prob)
    keep = rng.random(len(noisy)) > dropout
    keep = apply_sector_dropout(noisy, keep, config, rng)
    if keep.sum() < min(config.min_points, len(noisy)):
        target = min(config.min_points, len(noisy))
        rescued = rng.choice(len(noisy), size=target, replace=False)
        keep[rescued] = True
    noisy = noisy[keep]

    if noisy.shape[1] > 3 and config.intensity_std > 0:
        noisy[:, 3] += rng.normal(0.0, config.intensity_std, len(noisy)).astype(
            np.float32
        )
        noisy[:, 3] = np.clip(noisy[:, 3], 0.0, 1.0)

    outlier_count = int(round(len(noisy) * config.outlier_ratio))
    noisy = add_outliers(noisy, outlier_count, config, rng)
    rng.shuffle(noisy)

    return noisy.astype(np.float32), {
        "input_points": int(len(points)),
        "kept_points": int(keep.sum()),
        "outlier_points": int(outlier_count),
        "output_points": int(len(noisy)),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                json.dump(
                    record,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def scene_summary_from_manifest(
    scene_id: str,
    split: str,
    generation_fingerprint: str,
    map_classes: Sequence[str],
    records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    point_counts = [int(record["point_count"]) for record in records]
    per_class_sums = {name: 0 for name in map_classes}
    for record in records:
        for name, value in zip(map_classes, record["per_class_sums"]):
            per_class_sums[name] += int(value)
    return {
        "status": "complete",
        "scene_id": scene_id,
        "split": split,
        "generation_fingerprint": generation_fingerprint,
        "frame_count": len(records),
        "point_count": {
            "min": min(point_counts) if point_counts else 0,
            "max": max(point_counts) if point_counts else 0,
            "mean": float(sum(point_counts) / len(point_counts))
            if point_counts
            else 0.0,
        },
        "per_class_sums": per_class_sums,
        "observed_sum": sum(int(record["observed_sum"]) for record in records),
    }


def refresh_dataset_manifests_and_summaries(
    dst_root: Path,
    splits: Sequence[str],
    point_counts: Dict[str, int],
) -> None:
    metadata = load_json(dst_root / "dataset_metadata.json")
    split_payload = load_json(dst_root / "splits.json")
    map_classes = metadata["map_classes"]
    generation_fingerprint = metadata["generation_fingerprint"]
    updated_scene_summaries: Dict[str, Dict[str, object]] = {}

    for split in splits:
        for scene_id in split_payload[split]:
            scene_id = normalize_relative_path(scene_id)
            manifest_path = dst_root / scene_id / "manifest.jsonl"
            records = load_manifest(manifest_path)
            changed = False
            for record in records:
                lidar_path = normalize_relative_path(record["lidar_path"])
                if lidar_path in point_counts:
                    record["point_count"] = int(point_counts[lidar_path])
                    changed = True
            if not changed:
                continue
            atomic_jsonl(manifest_path, records)
            summary = scene_summary_from_manifest(
                scene_id,
                split,
                generation_fingerprint,
                map_classes,
                records,
            )
            atomic_json(dst_root / scene_id / "summary.json", summary)
            updated_scene_summaries[scene_id] = summary

    if updated_scene_summaries:
        root_summary_path = dst_root / "multi_scene_summary.json"
        root_summary = load_json(root_summary_path)
        root_summary["scene_summaries"] = [
            updated_scene_summaries.get(item["scene_id"], item)
            for item in root_summary["scene_summaries"]
        ]
        atomic_json(root_summary_path, root_summary)


def create_noisy_points(
    src_root: Path,
    dst_root: Path,
    point_paths: Sequence[str],
    load_dim: int,
    config: NoiseConfig,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    totals = {
        "files": 0,
        "input_points": 0,
        "kept_points": 0,
        "outlier_points": 0,
        "output_points": 0,
    }
    output_point_counts: Dict[str, int] = {}
    for index, rel_path in enumerate(point_paths, start=1):
        source = src_root / rel_path
        destination = dst_root / rel_path
        points = load_points(source, load_dim)
        noisy, stats = add_realistic_noise(points, config, rng)
        save_points(destination, noisy)
        output_point_counts[rel_path] = int(stats["output_points"])
        totals["files"] += 1
        for key in ("input_points", "kept_points", "outlier_points", "output_points"):
            totals[key] += stats[key]
        if index == 1 or index % 100 == 0 or index == len(point_paths):
            print(
                f"[noise] {index}/{len(point_paths)} {rel_path}: "
                f"{stats['input_points']} -> {stats['output_points']} points"
            )
    return totals, output_point_counts


def assert_safe_roots(src_root: Path, dst_root: Path) -> None:
    if src_root == dst_root:
        raise ValueError("dst-root must be different from src-root")
    if not src_root.is_dir():
        raise FileNotFoundError(f"src-root does not exist: {src_root}")
    if dst_root.exists() and any(dst_root.iterdir()):
        raise RuntimeError(
            f"dst-root already exists and is not empty: {dst_root}. "
            "Use a new output directory to avoid mixing experiments."
        )


def write_noise_metadata(
    dst_root: Path,
    args: argparse.Namespace,
    splits: Sequence[str],
    point_paths: Sequence[str],
    config: NoiseConfig,
    linked_files: int,
    totals: Dict[str, float],
) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.src_root.expanduser().resolve()),
        "destination_root": str(args.dst_root.expanduser().resolve()),
        "splits": list(splits),
        "profile": args.profile,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "load_dim": args.load_dim,
        "noise_config": asdict(config),
        "noisy_point_file_count": len(point_paths),
        "linked_or_copied_file_count": linked_files,
        "totals": totals,
    }
    with (dst_root / "noise_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    splits = selected_splits(args.splits)
    config = build_noise_config(args)

    assert_safe_roots(src_root, dst_root)
    point_paths = sorted(collect_point_paths(src_root, splits))
    print(f"[plan] source: {src_root}")
    print(f"[plan] destination: {dst_root}")
    print(f"[plan] noisy splits: {', '.join(splits)}")
    print(f"[plan] noisy point files: {len(point_paths)}")
    print(f"[plan] profile: {args.profile}")
    print(f"[plan] copy mode for unchanged files: {args.copy_mode}")
    if args.dry_run:
        return 0

    require_numpy()
    dst_root.mkdir(parents=True, exist_ok=True)
    linked_files = clone_dataset_shell(
        src_root, dst_root, set(point_paths), args.copy_mode
    )
    totals, point_counts = create_noisy_points(
        src_root, dst_root, point_paths, args.load_dim, config, args.seed
    )
    refresh_dataset_manifests_and_summaries(dst_root, splits, point_counts)
    write_noise_metadata(
        dst_root, args, splits, point_paths, config, linked_files, totals
    )
    print(f"[done] unchanged files materialized: {linked_files}")
    print(f"[done] noisy files written: {totals['files']}")
    print(f"[done] metadata: {dst_root / 'noise_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python tools/create_robot_bev_noisy_dataset.py \
#   --src-root data/replica_robot_bev_v4 \
#   --dst-root data/replica_robot_bev_v4_noisy_mid \
#   --splits test \
#   --profile light \
#   --seed 20260723 \
#   --copy-mode hardlink