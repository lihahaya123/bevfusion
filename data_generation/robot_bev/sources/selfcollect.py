"""Self-collected left-camera source adapter for canonical Robot BEV data."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..projection import (
    depth_to_base_points,
    dilate_binary,
    make_observation_mask,
    rasterize_point_mask,
    rasterize_semantics,
    semantic_boundary_mask,
)
from ..schema import BEV_XBOUND, BEV_YBOUND, BEV_ZBOUND, MAP_CLASSES
from ..writer import FramePayload, RobotBEVWriter


SOURCE_CLASS_NAMES = {
    0: "BLANKET",
    1: "CHAIR",
    2: "DOOR",
    3: "ELEVATOR",
    4: "FLOOR",
    5: "FLOOR_SIGN",
    6: "WASTE",
    7: "GLASS",
    8: "PERSON",
    9: "GLASS_FRAME",
    10: "STATION",
    11: "STEP",
    12: "SUNDRIES",
    13: "SWEEPER_STATION",
    14: "TURNSTILE",
    15: "WALL",
    16: "WIRE",
    17: "ESCALATOR",
    18: "COUCH",
    19: "HIK_AGV",
}

# Deliberately conservative: uncertain source categories are not forced into a
# canonical class. Their projected cells are removed from supervision.
DEFAULT_SEMANTIC_ID_TO_CLASS = {
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
DEFAULT_IGNORED_SEMANTIC_IDS = frozenset(
    set(SOURCE_CLASS_NAMES) - set(DEFAULT_SEMANTIC_ID_TO_CLASS)
) | {255}
LOSSY_SUFFIXES = frozenset({".jpg", ".jpeg"})
GENERATOR_VERSION = "1"
BEV_LABEL_SOURCE = "selfcollect_left_semantic_depth_projection"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SemanticPolicy:
    id_to_class: Mapping[int, str]
    ignore_ids: frozenset


@dataclass(frozen=True)
class SelfCollectFrame:
    raw_frame_id: int
    rgb_path: Path
    depth_path: Path
    label_path: Path


@dataclass(frozen=True)
class TrajectoryPose:
    raw_timestamp: float
    map_camera_from_camera: np.ndarray


def load_intrinsic(path: Path) -> np.ndarray:
    """Load fx, fy, cx, cy, accepting the source's trailing ``f`` suffix."""
    text = Path(path).read_text(encoding="utf-8")
    values = [
        float(value)
        for value in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text
        )
    ]
    if len(values) != 4:
        raise ValueError(
            f"expected exactly fx, fy, cx, cy in {path}; found {len(values)} values"
        )
    fx, fy, cx, cy = values
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"focal lengths must be positive in {path}")
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_semantic_policy(path: Optional[Path] = None) -> SemanticPolicy:
    if path is None:
        return SemanticPolicy(
            id_to_class=dict(DEFAULT_SEMANTIC_ID_TO_CLASS),
            ignore_ids=DEFAULT_IGNORED_SEMANTIC_IDS,
        )
    config_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read semantic map {config_path}") from exc
    raw_mapping = data.get("id_to_class")
    raw_ignore = data.get("ignore_ids", [])
    if not isinstance(raw_mapping, dict) or not isinstance(raw_ignore, list):
        raise ValueError("semantic map requires id_to_class object and ignore_ids list")
    mapping: Dict[int, str] = {}
    for raw_id, class_name in raw_mapping.items():
        semantic_id = int(raw_id)
        if semantic_id < 0 or semantic_id > np.iinfo(np.uint16).max:
            raise ValueError(f"semantic ID out of uint16 range: {semantic_id}")
        if class_name not in MAP_CLASSES:
            raise ValueError(f"unknown canonical class {class_name!r}")
        mapping[semantic_id] = str(class_name)
    ignore_ids = frozenset(int(value) for value in raw_ignore)
    overlap = set(mapping) & set(ignore_ids)
    if overlap:
        raise ValueError(f"semantic IDs cannot be mapped and ignored: {sorted(overlap)}")
    if not mapping:
        raise ValueError("semantic map must retain at least one source class")
    return SemanticPolicy(id_to_class=mapping, ignore_ids=ignore_ids)


def _collect_by_stem(directory: Path, suffixes: Sequence[str]) -> Dict[str, Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir() if directory.is_dir() else []):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.stem in files:
            raise ValueError(
                f"duplicate frame stem {path.stem!r} in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
        files[path.stem] = path
    return files


def _raw_frame_id(stem: str) -> int:
    match = re.match(r"^(\d+)", stem)
    if match is None:
        raise ValueError(f"frame name does not start with an integer ID: {stem!r}")
    return int(match.group(1))


def collect_frames(dataset_root: Path) -> List[SelfCollectFrame]:
    root = Path(dataset_root).expanduser().resolve()
    rgb = _collect_by_stem(root / "Left", (".jpg", ".jpeg", ".png"))
    depth = _collect_by_stem(root / "Depth", (".png",))
    labels = _collect_by_stem(root / "Label", (".jpg", ".jpeg", ".png"))
    if not rgb or not depth or not labels:
        raise FileNotFoundError(
            f"{root} must contain nonempty Left, Depth, and Label directories"
        )
    if set(rgb) != set(depth) or set(rgb) != set(labels):
        missing = {
            "missing_depth": sorted(set(rgb) - set(depth)),
            "missing_label": sorted(set(rgb) - set(labels)),
            "missing_rgb": sorted((set(depth) | set(labels)) - set(rgb)),
        }
        raise ValueError(f"Left/Depth/Label frame stems do not match: {missing}")
    frames = [
        SelfCollectFrame(
            raw_frame_id=_raw_frame_id(stem),
            rgb_path=rgb[stem],
            depth_path=depth[stem],
            label_path=labels[stem],
        )
        for stem in rgb
    ]
    frames.sort(key=lambda frame: frame.raw_frame_id)
    ids = [frame.raw_frame_id for frame in frames]
    if len(ids) != len(set(ids)):
        raise ValueError("more than one frame stem resolves to the same raw frame ID")
    return frames


def split_frames(
    frames: Sequence[SelfCollectFrame],
    *,
    split_mode: str,
    single_split: str = "train",
    split_ratios: Sequence[int] = (7, 1, 1),
) -> Dict[str, List[SelfCollectFrame]]:
    """Assign ordered frames to one split or periodic train/val/test slots."""
    assignments = {name: [] for name in SPLIT_NAMES}
    if split_mode == "single":
        if single_split not in SPLIT_NAMES:
            raise ValueError(f"unknown single split: {single_split!r}")
        assignments[single_split] = list(frames)
        return assignments
    if split_mode != "sampled":
        raise ValueError(f"unknown split mode: {split_mode!r}")
    if len(split_ratios) != len(SPLIT_NAMES):
        raise ValueError("split_ratios must contain train, val, and test counts")
    ratios = tuple(int(value) for value in split_ratios)
    if any(value <= 0 for value in ratios):
        raise ValueError("sampled split ratios must all be positive")
    cycle = sum(ratios)
    boundaries = np.cumsum(ratios)
    for index, frame in enumerate(frames):
        slot = index % cycle
        split_index = int(np.searchsorted(boundaries, slot, side="right"))
        assignments[SPLIT_NAMES[split_index]].append(frame)
    return assignments


def scene_ids_for_splits(
    scene: str,
    frame_splits: Mapping[str, Sequence[SelfCollectFrame]],
    *,
    split_mode: str,
) -> Dict[str, str]:
    if split_mode == "single":
        return {
            split: scene
            for split, split_frames_for_name in frame_splits.items()
            if split_frames_for_name
        }
    return {
        split: f"{scene}_{split}"
        for split, split_frames_for_name in frame_splits.items()
        if split_frames_for_name
    }


def _quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(value) for value in quaternion]
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("trajectory quaternion must be finite and nonzero")
    x, y, z, w = [value / norm for value in (x, y, z, w)]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def load_trajectory(
    path: Path, *, pose_convention: str = "camera_to_map"
) -> Dict[int, TrajectoryPose]:
    values = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"{path} must contain timestamp tx ty tz qx qy qz qw")
    if pose_convention not in {"camera_to_map", "map_to_camera"}:
        raise ValueError(f"unknown pose convention: {pose_convention}")
    poses: Dict[int, TrajectoryPose] = {}
    for row in values:
        raw_frame_id = int(round(float(row[0]) / 1_000_000.0))
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = _quaternion_xyzw_to_matrix(row[4:8])
        transform[:3, 3] = row[1:4].astype(np.float32)
        if pose_convention == "map_to_camera":
            transform = np.linalg.inv(transform).astype(np.float32)
        if raw_frame_id in poses:
            raise ValueError(f"duplicate trajectory frame ID {raw_frame_id} in {path}")
        poses[raw_frame_id] = TrajectoryPose(
            raw_timestamp=float(row[0]),
            map_camera_from_camera=transform,
        )
    return poses


def load_transform(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape == (3, 4):
        output = np.eye(4, dtype=np.float32)
        output[:3] = matrix
        matrix = output
    if matrix.shape != (4, 4):
        raise ValueError(f"{path} must contain a 3x4 or 4x4 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} contains nonfinite values")
    return matrix.astype(np.float32)


def _read_depth_mm(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise ValueError(f"depth must be a single-channel integer image: {path}")
    if depth.min() < 0 or depth.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"depth values do not fit uint16 millimetres: {path}")
    return depth.astype(np.uint16)


def _read_semantics(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        semantics = np.asarray(image)
    if semantics.ndim != 2 or not np.issubdtype(semantics.dtype, np.integer):
        raise ValueError(f"semantic label must be a single-channel integer image: {path}")
    if semantics.min() < 0 or semantics.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"semantic IDs do not fit uint16: {path}")
    return semantics.astype(np.uint16)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _validate_frame_shapes(frame: SelfCollectFrame) -> Tuple[int, int]:
    rgb = _read_rgb(frame.rgb_path)
    depth = _read_depth_mm(frame.depth_path)
    semantics = _read_semantics(frame.label_path)
    if rgb.shape[:2] != depth.shape or depth.shape != semantics.shape:
        raise ValueError(
            f"shape mismatch for raw frame {frame.raw_frame_id}: "
            f"rgb={rgb.shape[:2]} depth={depth.shape} label={semantics.shape}"
        )
    return depth.shape


def estimate_camera2base_from_floor(
    frames: Sequence[SelfCollectFrame],
    intrinsic: np.ndarray,
    *,
    floor_id: int = 4,
    boundary_radius: int = 1,
    sample_stride: int = 4,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Estimate fixed camera height/roll/pitch from labelled floor points."""
    samples: List[np.ndarray] = []
    identity = np.eye(4, dtype=np.float32)
    for frame in frames:
        depth = _read_depth_mm(frame.depth_path).astype(np.float32) * 0.001
        semantics = _read_semantics(frame.label_path)
        boundary = semantic_boundary_mask(semantics, boundary_radius).astype(bool)
        floor_mask = (semantics == int(floor_id)) & ~boundary
        points, _ = depth_to_base_points(
            depth,
            intrinsic,
            identity,
            pixel_mask=floor_mask,
            stride=sample_stride,
            max_depth=float("inf"),
        )
        if points.shape[0]:
            samples.append(points[:, :3])
    if not samples:
        raise RuntimeError(f"no usable FLOOR={floor_id} points for extrinsic estimation")
    points = np.concatenate(samples, axis=0).astype(np.float64)
    if points.shape[0] < 100:
        raise RuntimeError(
            f"only {points.shape[0]} floor points available; at least 100 required"
        )

    design = np.column_stack([points[:, 0], points[:, 2], np.ones(len(points))])
    target = points[:, 1]
    inliers = np.ones(len(points), dtype=bool)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(6):
        coefficients = np.linalg.lstsq(
            design[inliers], target[inliers], rcond=None
        )[0]
        residual = target - design @ coefficients
        median = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - median)))
        threshold = min(0.05, max(0.015, 4.0 * 1.4826 * mad))
        updated = np.abs(residual - median) <= threshold
        if updated.sum() < 100:
            raise RuntimeError("robust floor fit rejected too many points")
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    a, b, c = coefficients
    normal_down = np.asarray([-a, 1.0, -b], dtype=np.float64)
    normal_down /= np.linalg.norm(normal_down)
    up = -normal_down
    optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward = optical_forward - np.dot(optical_forward, up) * up
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    rotation = np.stack([forward, left, up], axis=0)
    height = abs(float(c)) / float(np.linalg.norm([-a, 1.0, -b]))
    camera2base = np.eye(4, dtype=np.float32)
    camera2base[:3, :3] = rotation.astype(np.float32)
    camera2base[:3, 3] = [0.0, 0.0, height]

    fitted = target[inliers] - design[inliers] @ coefficients
    diagnostics = {
        "method": "robust_floor_plane",
        "floor_id": int(floor_id),
        "point_count": int(len(points)),
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "height_m": float(height),
        "rmse_m": float(np.sqrt(np.mean(fitted * fitted))),
        "up_in_camera": up.tolist(),
    }
    return camera2base, diagnostics


def map_from_base(
    pose: TrajectoryPose, camera2base: np.ndarray
) -> np.ndarray:
    """Conjugate a camera trajectory into the canonical base/map frame."""
    camera2base = np.asarray(camera2base, dtype=np.float32)
    return (
        camera2base
        @ pose.map_camera_from_camera
        @ np.linalg.inv(camera2base)
    ).astype(np.float32)


def _trusted_pixel_mask(
    semantics: np.ndarray,
    policy: SemanticPolicy,
    boundary_radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mapped = np.isin(semantics, list(policy.id_to_class))
    boundary = semantic_boundary_mask(semantics, boundary_radius).astype(bool)
    trusted = mapped & ~boundary
    return trusted, ~trusted


def _build_frame_payload(
    output_frame_id: int,
    frame: SelfCollectFrame,
    args: argparse.Namespace,
    intrinsic: np.ndarray,
    policy: SemanticPolicy,
    camera2base: np.ndarray,
    pose: TrajectoryPose,
) -> Tuple[FramePayload, int, float, float]:
    rgb = _read_rgb(frame.rgb_path)
    depth_mm = _read_depth_mm(frame.depth_path)
    semantics = _read_semantics(frame.label_path)
    depth_m = depth_mm.astype(np.float32) * 0.001
    depth_valid = np.isfinite(depth_m) & (depth_m > 0.0)
    trusted_pixels, uncertain_pixels = _trusted_pixel_mask(
        semantics, policy, args.label_boundary_pixels
    )
    trusted_pixels &= depth_valid
    uncertain_pixels &= depth_valid

    points, _ = depth_to_base_points(
        depth_m,
        intrinsic,
        camera2base,
        stride=args.depth_stride,
        max_depth=args.max_depth,
        max_points=args.max_points,
    )
    label_points, semantic_ids = depth_to_base_points(
        depth_m,
        intrinsic,
        camera2base,
        pixel_mask=trusted_pixels,
        stride=1,
        max_depth=args.max_depth,
        semantic=semantics,
    )
    observation_points, _ = depth_to_base_points(
        depth_m,
        intrinsic,
        camera2base,
        pixel_mask=trusted_pixels,
        stride=1,
        max_depth=float("inf"),
    )
    uncertain_points, _ = depth_to_base_points(
        depth_m,
        intrinsic,
        camera2base,
        pixel_mask=uncertain_pixels,
        stride=1,
        max_depth=float("inf"),
    )
    if points.shape[0] < args.min_points:
        raise RuntimeError(
            f"raw frame {frame.raw_frame_id} has only {points.shape[0]} points"
        )

    observed = make_observation_mask(
        observation_points,
        camera2base[:3, 3],
        BEV_XBOUND,
        BEV_YBOUND,
    )
    ignored_bev = rasterize_point_mask(
        uncertain_points, BEV_XBOUND, BEV_YBOUND, BEV_ZBOUND
    )
    ignored_bev = dilate_binary(ignored_bev, args.ignore_dilation_cells)
    observed[ignored_bev.astype(bool)] = 0
    labels = rasterize_semantics(
        label_points,
        (
            semantic_ids
            if semantic_ids is not None
            else np.zeros((0,), dtype=np.int64)
        ),
        policy.id_to_class,
        BEV_XBOUND,
        BEV_YBOUND,
        BEV_ZBOUND,
    )
    labels *= observed[None]

    semantic_coverage = float(trusted_pixels.sum() / max(1, depth_valid.sum()))
    observed_coverage = float(observed.mean())
    if semantic_coverage < args.min_semantic_coverage:
        raise RuntimeError(
            f"raw frame {frame.raw_frame_id} semantic coverage "
            f"{semantic_coverage:.6f} below {args.min_semantic_coverage:.6f}"
        )
    if observed_coverage < args.min_observed_coverage:
        raise RuntimeError(
            f"raw frame {frame.raw_frame_id} observed coverage "
            f"{observed_coverage:.6f} below {args.min_observed_coverage:.6f}"
        )

    timestamp_us = int(round(pose.raw_timestamp / 1000.0))
    payload = FramePayload(
        frame_id=output_frame_id,
        timestamp=timestamp_us,
        rgb=rgb,
        points=points.astype(np.float32, copy=False),
        bev_labels=labels.astype(np.uint8, copy=False),
        observed_mask=observed.astype(np.uint8, copy=False),
        class_validity=np.ones((len(MAP_CLASSES),), dtype=np.uint8),
        cam_intrinsic=intrinsic,
        camera2base=camera2base,
        lidar2base=np.eye(4, dtype=np.float32),
        map_from_base=map_from_base(pose, camera2base),
        depth_mm=depth_mm,
        semantics=semantics.astype(np.uint16, copy=False),
        extra_info={
            "raw_frame_id": int(frame.raw_frame_id),
            "semantic_coverage": semantic_coverage,
            "observed_coverage": observed_coverage,
            "ignored_bev_coverage": float(ignored_bev.mean()),
            "lossy_label": frame.label_path.suffix.lower() in LOSSY_SUFFIXES,
        },
    )
    return payload, points.shape[0], semantic_coverage, observed_coverage


def _generation_parameters(
    args: argparse.Namespace,
    frames: Sequence[SelfCollectFrame],
    frame_splits: Mapping[str, Sequence[SelfCollectFrame]],
    scene_ids: Mapping[str, str],
    policy: SemanticPolicy,
    camera2base: np.ndarray,
    extrinsic_diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "scene_prefix": args.scene,
        "split_mode": args.split_mode,
        "single_split": args.split,
        "split_ratios": list(args.split_ratios),
        "split_scene_ids": dict(scene_ids),
        "split_frame_ids": {
            split: [frame.raw_frame_id for frame in split_frames_for_name]
            for split, split_frames_for_name in frame_splits.items()
        },
        "frame_ids": [frame.raw_frame_id for frame in frames],
        "intrinsic_file": "in.txt",
        "trajectory_file": args.trajectory_file,
        "pose_convention": args.pose_convention,
        "timestamp_source_unit": "nanoseconds",
        "depth_unit": "millimetres",
        "depth_type": "opencv_z_depth",
        "semantic_id_to_class": {
            str(key): value for key, value in sorted(policy.id_to_class.items())
        },
        "ignored_semantic_ids": sorted(int(value) for value in policy.ignore_ids),
        "source_class_names": {
            str(key): value for key, value in SOURCE_CLASS_NAMES.items()
        },
        "bev_label_source": BEV_LABEL_SOURCE,
        "uncertain_region_policy": "invalid_overrides_observed",
        "lossy_semantic_input": any(
            frame.label_path.suffix.lower() in LOSSY_SUFFIXES for frame in frames
        ),
        "allow_lossy_labels": bool(args.allow_lossy_labels),
        "label_boundary_pixels": int(args.label_boundary_pixels),
        "ignore_dilation_cells": int(args.ignore_dilation_cells),
        "xbound": list(BEV_XBOUND),
        "ybound": list(BEV_YBOUND),
        "zbound": list(BEV_ZBOUND),
        "max_depth": float(args.max_depth),
        "depth_stride": int(args.depth_stride),
        "max_points": int(args.max_points),
        "camera2base": np.asarray(camera2base).tolist(),
        "camera2base_source": (
            str(Path(args.camera2base).expanduser().resolve())
            if args.camera2base is not None
            else "estimated_from_floor"
        ),
        "camera2base_diagnostics": dict(extrinsic_diagnostics),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.depth_stride <= 0 or args.max_points <= 0 or args.min_points <= 0:
        raise ValueError("depth stride and point thresholds must be positive")
    if args.max_depth <= 0.0:
        raise ValueError("max depth must be positive")
    if args.label_boundary_pixels < 0 or args.ignore_dilation_cells < 0:
        raise ValueError("mask dilation values must be non-negative")
    for name in ("min_semantic_coverage", "min_observed_coverage"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if args.split_mode == "sampled":
        if len(args.split_ratios) != 3 or any(
            int(value) <= 0 for value in args.split_ratios
        ):
            raise ValueError(
                "sampled --split-ratios requires three positive integers"
            )


def run_generation(args: argparse.Namespace) -> None:
    _validate_args(args)
    dataset_root = Path(args.dataset).expanduser().resolve()
    frames = collect_frames(dataset_root)
    frame_splits = split_frames(
        frames,
        split_mode=args.split_mode,
        single_split=args.split,
        split_ratios=args.split_ratios,
    )
    scene_ids = scene_ids_for_splits(
        args.scene, frame_splits, split_mode=args.split_mode
    )
    intrinsic = load_intrinsic(dataset_root / "in.txt")
    policy = load_semantic_policy(
        Path(args.semantic_map) if args.semantic_map is not None else None
    )
    trajectory = load_trajectory(
        dataset_root / args.trajectory_file,
        pose_convention=args.pose_convention,
    )
    missing_poses = [
        frame.raw_frame_id for frame in frames if frame.raw_frame_id not in trajectory
    ]
    if missing_poses:
        raise ValueError(f"frames missing from trajectory: {missing_poses}")
    shapes = {_validate_frame_shapes(frame) for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"all frames must share one image shape; found {sorted(shapes)}")
    lossy_frames = [
        frame.label_path.name
        for frame in frames
        if frame.label_path.suffix.lower() in LOSSY_SUFFIXES
    ]
    if lossy_frames and not args.allow_lossy_labels:
        raise ValueError(
            "lossy semantic JPEG input is disabled; provide PNG labels or pass "
            "--allow-lossy-labels for diagnostic generation"
        )

    if args.camera2base is not None:
        camera2base = load_transform(Path(args.camera2base))
        extrinsic_diagnostics: Mapping[str, object] = {"method": "provided"}
    else:
        camera2base, extrinsic_diagnostics = estimate_camera2base_from_floor(
            frames,
            intrinsic,
            floor_id=args.floor_id,
            boundary_radius=args.label_boundary_pixels,
            sample_stride=args.floor_sample_stride,
        )
    preflight = {
        "status": "preflight_ok",
        "dataset": str(dataset_root),
        "scene": args.scene,
        "frame_count": len(frames),
        "split_mode": args.split_mode,
        "split_ratios": list(args.split_ratios),
        "split_frame_counts": {
            split: len(split_frames_for_name)
            for split, split_frames_for_name in frame_splits.items()
        },
        "split_scene_ids": scene_ids,
        "image_shape": list(next(iter(shapes))),
        "lossy_label_count": len(lossy_frames),
        "camera2base": camera2base.tolist(),
        "camera2base_diagnostics": dict(extrinsic_diagnostics),
    }
    print(json.dumps(preflight, sort_keys=True))
    if args.preflight_only:
        return

    splits = {
        split: [scene_ids[split]] if split in scene_ids else []
        for split in SPLIT_NAMES
    }
    writer = RobotBEVWriter(
        root=Path(args.output_dir),
        dataset_id=args.dataset_id,
        source_type="real",
        source_dataset="selfcollect_left",
        generator_name="selfcollect_robot_bev",
        generator_version=GENERATOR_VERSION,
        splits=splits,
        generation_parameters=_generation_parameters(
            args,
            frames,
            frame_splits,
            scene_ids,
            policy,
            camera2base,
            extrinsic_diagnostics,
        ),
        resume=args.resume,
    )
    for split in SPLIT_NAMES:
        split_frames_for_name = frame_splits[split]
        if not split_frames_for_name:
            continue
        scene_id = scene_ids[split]
        manifest_path = writer.root / scene_id / "manifest.jsonl"
        completed = 0
        if args.resume and manifest_path.is_file():
            completed = sum(
                1 for line in manifest_path.read_text().splitlines() if line
            )
            if completed > len(split_frames_for_name):
                raise RuntimeError(
                    f"existing {scene_id} manifest has more frames than its split"
                )

        for output_frame_id, frame in enumerate(
            split_frames_for_name[completed:], start=completed
        ):
            payload, point_count, semantic_coverage, observed_coverage = (
                _build_frame_payload(
                    output_frame_id,
                    frame,
                    args,
                    intrinsic,
                    policy,
                    camera2base,
                    trajectory[frame.raw_frame_id],
                )
            )
            writer.write_frame(scene_id, split, payload)
            print(
                f"[{split} {output_frame_id + 1:06d}/"
                f"{len(split_frames_for_name):06d}] "
                f"raw={frame.raw_frame_id} points={point_count} "
                f"semantic={semantic_coverage:.3f} "
                f"observed={observed_coverage:.3f}"
            )
        writer.finalize_scene(scene_id, split)
    writer.finalize_dataset()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical RobotBEV v4 labels from self-collected left RGB, "
            "depth, and two-dimensional semantics."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scene", default="selfcollect_001")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--split-mode",
        choices=("single", "sampled"),
        default="single",
        help=(
            "single writes every frame to --split; sampled periodically assigns "
            "ordered frames according to --split-ratios"
        ),
    )
    parser.add_argument(
        "--split-ratios",
        type=int,
        nargs=3,
        default=(7, 1, 1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="periodic train/val/test frame counts used by sampled mode",
    )
    parser.add_argument("--output-dir", default="data/selfcollect_robot_bev")
    parser.add_argument("--semantic-map")
    parser.add_argument("--camera2base")
    parser.add_argument("--trajectory-file", default="CameraTrajectory.txt")
    parser.add_argument(
        "--pose-convention",
        choices=("camera_to_map", "map_to_camera"),
        default="camera_to_map",
    )
    parser.add_argument("--floor-id", type=int, default=4)
    parser.add_argument("--floor-sample-stride", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--min-semantic-coverage", type=float, default=0.30)
    parser.add_argument("--min-observed-coverage", type=float, default=0.01)
    parser.add_argument("--label-boundary-pixels", type=int, default=1)
    parser.add_argument("--ignore-dilation-cells", type=int, default=0)
    parser.add_argument("--allow-lossy-labels", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


__all__ = [
    "BEV_LABEL_SOURCE",
    "DEFAULT_IGNORED_SEMANTIC_IDS",
    "DEFAULT_SEMANTIC_ID_TO_CLASS",
    "SemanticPolicy",
    "SelfCollectFrame",
    "SPLIT_NAMES",
    "collect_frames",
    "estimate_camera2base_from_floor",
    "load_intrinsic",
    "load_semantic_policy",
    "load_trajectory",
    "make_parser",
    "map_from_base",
    "run_generation",
    "scene_ids_for_splits",
    "split_frames",
]
