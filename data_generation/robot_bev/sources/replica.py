"""Original Replica v1 PTex source adapter for canonical robot BEV data."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import BEV_XBOUND, BEV_YBOUND, MAP_CLASSES
from ..writer import FramePayload, RobotBEVWriter
from . import habitat_common
from .habitat_common import (
    DEPTH_UUID,
    RGB_UUID,
    SEMANTIC_UUID,
    base_grid_to_habitat_local,
    camera_optical_to_base_matrix,
    depth_to_points,
    initialize_agent,
    initialize_navmesh,
    is_floor_level_safe,
    make_camera_intrinsic,
    make_cfg,
    make_observation_mask,
    map_from_base_matrix,
    next_action,
    sensor_to_base_matrix,
    transform_habitat_local_to_world,
    turn_agent_away,
)


FURNITURE_CATEGORIES = frozenset(
    {
        "base cabinet",
        "beanbag",
        "bed",
        "bench",
        "cabinet",
        "chair",
        "desk",
        "nightstand",
        "plant stand",
        "rack",
        "shelf",
        "sofa",
        "stool",
        "table",
        "tv stand",
        "wall cabinet",
    }
)
SEMANTIC_CATEGORY_GROUPS = {
    "carpet": frozenset({"carpet", "floor mat", "mat", "rug"}),
    "wall": frozenset({"wall"}),
    "floor": frozenset({"floor", "ground"}),
    "furniture": FURNITURE_CATEGORIES,
}
IGNORED_SEMANTIC_CATEGORIES = frozenset(
    {"", "background", "none", "undefined", "unknown", "void"}
)
FRL_PTEX_ASSET_TYPE = 3
REQUIRED_HABITAT_VERSION = "0.2.2"
RESUME_POSE_ATOL = 1e-5
SIMULATION_TIMESTEP_SECONDS = 1.0 / 60.0


@dataclass(frozen=True)
class ReplicaSceneFiles:
    dataset_config: Path
    scene_dir: Path
    stage_config: Path
    render_mesh: Path
    semantic_mesh: Path
    semantic_descriptor: Path
    navmesh: Path
    ptex_parameters: Path
    ptex_atlases: Tuple[Path, ...]

    @property
    def ptex_atlas_count(self) -> int:
        return len(self.ptex_atlases)


@dataclass(frozen=True)
class NavmeshTopdown:
    grid: np.ndarray
    min_x: float
    min_z: float
    meters_per_pixel: float


@dataclass(frozen=True)
class TrajectoryState:
    last_collided: bool = False
    last_stair_recovery: bool = False
    stair_recoveries: int = 0


@dataclass(frozen=True)
class TrajectoryTransition:
    observations: Optional[Dict[str, object]]
    trajectory: TrajectoryState


def parse_bound(
    values: Sequence[float], name: str
) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: min max step")
    lo, hi, step = values
    if hi <= lo or step <= 0:
        raise ValueError(f"{name} must satisfy max > min and step > 0")
    return float(lo), float(hi), float(step)


def validate_replica_scene(
    dataset_config: Path, scene: str
) -> ReplicaSceneFiles:
    """Validate the official Replica v1 scene layout required by PTex."""
    dataset_config = Path(dataset_config).expanduser().resolve()
    if not dataset_config.is_file():
        raise FileNotFoundError(
            f"Replica dataset config does not exist: {dataset_config}"
        )
    if dataset_config.name != "replica.scene_dataset_config.json":
        raise RuntimeError(
            "This generator only accepts the original Replica dataset config named "
            "replica.scene_dataset_config.json (ReplicaCAD is intentionally unsupported)."
        )

    scene_dir = dataset_config.parent / scene
    habitat_dir = scene_dir / "habitat"
    stage_config = habitat_dir / "replica_stage.stage_config.json"
    expected = {
        "render mesh": scene_dir / "mesh.ply",
        "PTex parameters": scene_dir / "textures" / "parameters.json",
        "PTex sorted faces": habitat_dir / "sorted_faces.bin",
        "semantic mesh": habitat_dir / "mesh_semantic.ply",
        "semantic descriptor": habitat_dir / "info_semantic.json",
        "navmesh": habitat_dir / "mesh_semantic.navmesh",
        "stage config": stage_config,
    }
    missing = [
        f"{label}: {path}"
        for label, path in expected.items()
        if not path.is_file()
    ]
    atlases = tuple(
        sorted((scene_dir / "textures").glob("*-color-ptex.hdr"))
    )
    if not atlases:
        missing.append(
            f"PTex atlases: {scene_dir / 'textures' / '*-color-ptex.hdr'}"
        )
    if missing:
        raise FileNotFoundError(
            f"Scene {scene!r} is not a complete original Replica PTex scene:\n  "
            + "\n  ".join(missing)
        )

    try:
        stage_data = json.loads(stage_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot parse Replica stage config: {stage_config}"
        ) from exc
    descriptor = stage_data.get("semantic_descriptor_filename")
    if descriptor != "info_semantic.json":
        raise RuntimeError(
            f"{stage_config} must contain semantic_descriptor_filename="
            '"info_semantic.json" for original Replica semantics; '
            f"found {descriptor!r}."
        )
    if stage_data.get("render_asset") != "../mesh.ply":
        raise RuntimeError(
            f"Unexpected render_asset in {stage_config}; expected '../mesh.ply' for PTex."
        )
    if stage_data.get("semantic_asset") != "mesh_semantic.ply":
        raise RuntimeError(
            f"Unexpected semantic_asset in {stage_config}; "
            "expected 'mesh_semantic.ply'."
        )
    if stage_data.get("nav_asset") != "mesh_semantic.navmesh":
        raise RuntimeError(
            f"Unexpected nav_asset in {stage_config}; "
            "expected 'mesh_semantic.navmesh'."
        )

    return ReplicaSceneFiles(
        dataset_config=dataset_config,
        scene_dir=scene_dir,
        stage_config=stage_config,
        render_mesh=expected["render mesh"],
        semantic_mesh=expected["semantic mesh"],
        semantic_descriptor=expected["semantic descriptor"],
        navmesh=expected["navmesh"],
        ptex_parameters=expected["PTex parameters"],
        ptex_atlases=atlases,
    )


def load_scene_splits(
    split_file: Optional[Path], scenes: Sequence[str]
) -> Dict[str, str]:
    """Return scene -> split, rejecting cross-split leakage."""
    if split_file is None:
        return {scene: "train" for scene in scenes}
    split_path = Path(split_file).expanduser().resolve()
    try:
        data = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read split JSON: {split_path}") from exc
    allowed = {"train", "val", "test"}
    unknown_keys = set(data) - allowed
    if unknown_keys:
        raise ValueError(
            f"Unknown split names in {split_path}: {sorted(unknown_keys)}"
        )

    assignments: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        values = data.get(split, [])
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise ValueError(
                f"Split {split!r} must be a JSON list of scene names"
            )
        for scene in values:
            if scene in assignments:
                raise ValueError(
                    f"Scene {scene!r} appears in more than one split: "
                    f"{assignments[scene]} and {split}"
                )
            assignments[scene] = split
    missing = sorted(set(scenes) - set(assignments))
    if missing:
        raise ValueError(
            f"Requested scenes missing from split file: {missing}"
        )
    return {scene: assignments[scene] for scene in scenes}


def semantic_category_to_map_class(
    category_name: str,
) -> Optional[str]:
    normalized = " ".join(
        category_name.lower().replace("_", " ").replace("-", " ").split()
    )
    if normalized in IGNORED_SEMANTIC_CATEGORIES:
        return None
    for map_class, categories in SEMANTIC_CATEGORY_GROUPS.items():
        if normalized in categories:
            return map_class
    return "other"


def build_semantic_id_to_class(sim) -> Dict[int, str]:
    habitat_common.require_habitat_sim()
    mapping: Dict[int, str] = {}
    scene = getattr(sim, "semantic_scene", None)
    objects = getattr(scene, "objects", None)
    if not objects:
        return mapping

    for fallback_id, obj in enumerate(objects):
        if obj is None:
            continue
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        semantic_id = fallback_id
        object_id = str(getattr(obj, "id", ""))
        match = re.search(r"(\d+)$", object_id)
        if match:
            semantic_id = int(match.group(1))
        map_class = semantic_category_to_map_class(category_name)
        if map_class is not None:
            mapping[int(semantic_id)] = map_class
    return mapping


def point_indices(
    points: np.ndarray,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    x = points[:, 0]
    y = points[:, 1]
    valid = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    height = int(round((x_max - x_min) / x_step))
    width = int(round((y_max - y_min) / y_step))
    rows = np.floor((x[valid] - x_min) / x_step).astype(np.int64)
    cols = np.floor((y[valid] - y_min) / y_step).astype(np.int64)
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    return rows, cols


def build_navmesh_topdown(
    sim, meters_per_pixel: float, height: float
) -> NavmeshTopdown:
    habitat_common.require_habitat_sim()
    bounds = sim.pathfinder.get_bounds()
    lower = np.asarray(bounds[0], dtype=np.float32)
    upper = np.asarray(bounds[1], dtype=np.float32)
    grid = np.asarray(
        sim.pathfinder.get_topdown_view(meters_per_pixel, float(height)),
        dtype=np.uint8,
    )
    return NavmeshTopdown(
        grid=grid,
        min_x=float(min(lower[0], upper[0])),
        min_z=float(min(lower[2], upper[2])),
        meters_per_pixel=float(meters_per_pixel),
    )


def sample_navmesh_topdown(
    cache: NavmeshTopdown, world: np.ndarray
) -> np.ndarray:
    world64 = np.asarray(world, dtype=np.float64)
    x_offset = world64[:, 0] - cache.min_x
    z_offset = world64[:, 2] - cache.min_z
    inside = (
        (z_offset >= 0.0)
        & (z_offset < cache.grid.shape[0] * cache.meters_per_pixel)
        & (x_offset >= 0.0)
        & (x_offset < cache.grid.shape[1] * cache.meters_per_pixel)
    )
    values = np.zeros(world.shape[0], dtype=np.uint8)
    if np.any(inside):
        rows = np.floor(
            z_offset[inside] / cache.meters_per_pixel
        ).astype(np.int64)
        cols = np.floor(
            x_offset[inside] / cache.meters_per_pixel
        ).astype(np.int64)
        rows = np.clip(rows, 0, cache.grid.shape[0] - 1)
        cols = np.clip(cols, 0, cache.grid.shape[1] - 1)
        values[inside] = cache.grid[rows, cols]
    return values


def make_bev_labels(
    sim,
    state,
    views: Sequence[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    semantic_id_to_class: Dict[int, str],
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    min_obstacle_height: float,
    max_obstacle_height: float,
    valid_mask: np.ndarray,
    navmesh_topdown: Optional[NavmeshTopdown] = None,
) -> np.ndarray:
    habitat_common.require_habitat_sim()
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    height = int(round((x_max - x_min) / x_step))
    width = int(round((y_max - y_min) / y_step))
    valid_mask = np.asarray(valid_mask, dtype=np.uint8)
    if valid_mask.shape != (height, width):
        raise ValueError(
            f"valid_mask must have shape {(height, width)}, "
            f"found {valid_mask.shape}"
        )
    mask = np.zeros((len(MAP_CLASSES), height, width), dtype=np.uint8)

    x_centers = x_min + (
        np.arange(height, dtype=np.float32) + 0.5
    ) * x_step
    y_centers = y_min + (
        np.arange(width, dtype=np.float32) + 0.5
    ) * y_step
    y_grid, x_grid = np.meshgrid(y_centers, x_centers)
    local = base_grid_to_habitat_local(x_grid, y_grid)
    world = transform_habitat_local_to_world(state, local)

    if navmesh_topdown is not None:
        floor = sample_navmesh_topdown(navmesh_topdown, world)
    else:
        floor = np.zeros((height * width,), dtype=np.uint8)
        for idx, point in enumerate(world):
            floor[idx] = (
                1
                if sim.pathfinder.is_navigable(point, max_y_delta=0.5)
                else 0
            )
    mask[MAP_CLASSES.index("floor")] = (
        floor.reshape(height, width).astype(bool)
        & valid_mask.astype(bool)
    ).astype(np.uint8)

    for points, semantic_ids, _ in views:
        if points.shape[0] == 0:
            continue
        obstacle_points = points[
            (points[:, 2] >= min_obstacle_height)
            & (points[:, 2] <= max_obstacle_height)
        ]
        if obstacle_points.shape[0] > 0:
            rows, cols = point_indices(obstacle_points, xbound, ybound)
            mask[MAP_CLASSES.index("obstacle"), rows, cols] = 1

        if (
            semantic_ids is None
            or not semantic_id_to_class
            or points.shape[0] != semantic_ids.shape[0]
        ):
            continue
        for semantic_id in np.unique(semantic_ids):
            map_class = semantic_id_to_class.get(int(semantic_id))
            if (
                map_class not in MAP_CLASSES
                or map_class in {"floor", "obstacle"}
            ):
                continue
            semantic_points = points[semantic_ids == semantic_id]
            if semantic_points.shape[0] == 0:
                continue
            rows, cols = point_indices(semantic_points, xbound, ybound)
            mask[MAP_CLASSES.index(map_class), rows, cols] = 1

    mask *= valid_mask[None]
    return mask


def load_scene_list(args: argparse.Namespace) -> List[str]:
    scenes: List[str] = []
    if args.scenes:
        scenes.extend(args.scenes)
    if args.scenes_file:
        scene_file = Path(args.scenes_file)
        for line in scene_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                scenes.append(stripped)
    if not scenes:
        scenes = [args.scene]
    deduped = []
    seen = set()
    for scene in scenes:
        if scene not in seen:
            deduped.append(scene)
            seen.add(scene)
    return deduped


def _jsonable_parameter(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_jsonable_parameter(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_parameter(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_parameter(item)
            for key, item in value.items()
        }
    return value


def _split_file_contents(split_file: Optional[str]):
    if not split_file:
        return None
    split_path = Path(split_file).expanduser().resolve()
    try:
        return json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read split JSON: {split_path}") from exc


def generation_parameters(args: argparse.Namespace) -> Dict[str, object]:
    """Return the complete training-artifact generation contract."""
    excluded = {
        "output_dir",
        "resume",
        "preflight_only",
        "gpu_id",
        "save_visualization",
        "save_ply",
    }
    parameters = {
        key: _jsonable_parameter(value)
        for key, value in vars(args).items()
        if key not in excluded
    }
    requested_scenes = load_scene_list(args)
    split_contents = _split_file_contents(args.split_file)
    if args.scenes_file:
        parameters["scenes_file"] = requested_scenes
    if args.split_file:
        parameters["split_file"] = split_contents
    parameters.update(
        {
            "requested_scenes": requested_scenes,
            "split_file_contents": split_contents,
            "use_physics": bool(
                args.enable_physics and not args.disable_physics
            ),
            "habitat_sim_version": str(
                getattr(habitat_common.habitat_sim, "__version__", "unavailable")
            ),
            "semantic_category_groups": {
                name: sorted(categories)
                for name, categories in SEMANTIC_CATEGORY_GROUPS.items()
            },
            "ignored_semantic_categories": sorted(
                IGNORED_SEMANTIC_CATEGORIES
            ),
        }
    )
    return parameters


def _require_replica_habitat(
    args: argparse.Namespace, *, rendering: bool
):
    """Enforce the Replica PTex Habitat version at every public entry point."""
    hs = habitat_common.require_habitat_sim()
    version = str(getattr(hs, "__version__", "unknown"))
    if version == REQUIRED_HABITAT_VERSION:
        return hs
    diagnostic_override = (
        not rendering
        and bool(getattr(args, "preflight_only", False))
        and bool(getattr(args, "allow_version_mismatch", False))
    )
    if diagnostic_override:
        return hs
    if rendering:
        raise RuntimeError(
            "formal Replica generation requires Habitat-Sim 0.2.2; "
            f"found {version}. Activate the habitat022 environment"
        )
    raise RuntimeError(
        "Original Replica PTex preflight requires Habitat-Sim 0.2.2; "
        f"found {version}. Use --allow-version-mismatch only with "
        "--preflight-only for non-production diagnostics"
    )


def _reject_unsupported_visualization_options(
    args: argparse.Namespace,
) -> None:
    unsupported = [
        option
        for option, enabled in (
            (
                "--save-visualization",
                bool(getattr(args, "save_visualization", False)),
            ),
            ("--save-ply", bool(getattr(args, "save_ply", False))),
        )
        if enabled
    ]
    if unsupported:
        raise ValueError(
            f"{', '.join(unsupported)} are deprecated and unsupported by "
            "the canonical writer; remove these options"
        )


def _validate_args(args: argparse.Namespace) -> None:
    _reject_unsupported_visualization_options(args)
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.depth_stride <= 0:
        raise ValueError("--depth-stride must be positive")
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    if min(args.width, args.height) <= 0:
        raise ValueError("Sensor resolution must be positive")
    if not 0.0 < args.hfov < 180.0:
        raise ValueError("Camera HFOV must be between 0 and 180 degrees")
    if args.agent_max_climb < 0:
        raise ValueError("--agent-max-climb must be non-negative")
    if args.agent_max_slope < 0:
        raise ValueError("--agent-max-slope must be non-negative")
    if args.navmesh_cell_size <= 0.0 or args.navmesh_cell_height <= 0.0:
        raise ValueError("Navmesh cell size and cell height must be positive")
    if 0.0 < args.agent_max_climb < args.navmesh_cell_height:
        raise ValueError(
            "--agent-max-climb is smaller than --navmesh-cell-height and would be "
            "quantized to zero by Recast; decrease --navmesh-cell-height"
        )
    if args.stair_check_radius <= 0:
        raise ValueError("--stair-check-radius must be positive")
    if args.max_floor_height_delta < 0:
        raise ValueError("--max-floor-height-delta must be non-negative")
    if args.safe_point_max_tries <= 0:
        raise ValueError("--safe-point-max-tries must be positive")
    if args.enable_physics and args.disable_physics:
        raise ValueError(
            "--enable-physics and --disable-physics are mutually exclusive"
        )
    xbound = parse_bound(args.xbound, "xbound")
    ybound = parse_bound(args.ybound, "ybound")
    if xbound != tuple(BEV_XBOUND) or ybound != tuple(BEV_YBOUND):
        raise ValueError(
            "--xbound and --ybound must match the fixed canonical schema: "
            f"xbound={BEV_XBOUND}, ybound={BEV_YBOUND}"
        )
    args.use_physics = bool(
        args.enable_physics and not args.disable_physics
    )


def _load_canonical_manifest(writer: RobotBEVWriter, scene: str):
    manifest_path = Path(writer.root) / scene / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(
                    f"Blank canonical manifest line at {manifest_path}:{line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid manifest JSON at {manifest_path}:{line_number}"
                ) from exc
            expected_frame = len(records)
            if record.get("frame_id") != expected_frame:
                raise RuntimeError(
                    f"Non-contiguous manifest at {manifest_path}:{line_number}; "
                    f"expected frame_id={expected_frame}, "
                    f"found {record.get('frame_id')!r}"
                )
            for key in (
                "image_path",
                "lidar_path",
                "bev_mask_path",
                "bev_observed_mask_path",
                "depth_path",
                "semantic_path",
            ):
                relative = record.get(key)
                if not relative or not (Path(writer.root) / str(relative)).is_file():
                    raise RuntimeError(
                        f"Manifest references missing {key}: {relative!r}"
                    )
            records.append(record)
    return records


def _advance_trajectory(
    sim,
    args: argparse.Namespace,
    frame_idx: int,
    trajectory: TrajectoryState,
    *,
    render_observations: bool,
) -> TrajectoryTransition:
    """Apply one deterministic post-frame-zero navigation transition."""
    if frame_idx <= 0:
        raise ValueError("trajectory transitions start at frame 1")
    frame_rng = random.Random(
        (args.seed + 1) * 1_000_003 + frame_idx
    )
    previous_state = sim.get_agent(0).get_state()
    action = next_action(trajectory.last_collided, frame_rng)
    if render_observations:
        observations = sim.step(action)
        collided = bool(observations.get("collided", False))
    else:
        collided = bool(sim.get_agent(0).act(action))
        sim.step_world(SIMULATION_TIMESTEP_SECONDS)
        observations = None

    current_state = sim.get_agent(0).get_state()
    stair_recovery = False
    if args.enable_stair_filter and not is_floor_level_safe(
        sim,
        current_state.position,
        args.stair_check_radius,
        args.max_floor_height_delta,
    ):
        turn_agent_away(sim, previous_state, frame_rng)
        if render_observations:
            observations = sim.get_sensor_observations()
        collided = True
        stair_recovery = True

    return TrajectoryTransition(
        observations=observations,
        trajectory=TrajectoryState(
            last_collided=collided,
            last_stair_recovery=stair_recovery,
            stair_recoveries=(
                trajectory.stair_recoveries + int(stair_recovery)
            ),
        ),
    )


def _validate_replayed_frame(
    args: argparse.Namespace,
    frame_idx: int,
    record: Dict[str, object],
    state,
) -> None:
    record_frame_id = record.get("frame_id")
    if record_frame_id != frame_idx:
        raise RuntimeError(
            f"Resume replay manifest mismatch for scene {args.scene!r} "
            f"at frame {frame_idx:06d}: frame_id={record_frame_id!r}, "
            f"expected {frame_idx}"
        )
    expected_timestamp = (
        args.timestamp_start + frame_idx * args.timestamp_step
    )
    record_timestamp = record.get("timestamp")
    if record_timestamp != expected_timestamp:
        raise RuntimeError(
            f"Resume replay manifest mismatch for scene {args.scene!r} "
            f"at frame {frame_idx:06d}: timestamp={record_timestamp!r}, "
            f"expected {expected_timestamp}"
        )

    expected_pose = np.asarray(record.get("T_map_base"), dtype=np.float32)
    if expected_pose.shape != (4, 4):
        raise RuntimeError(
            f"Resume replay manifest mismatch for scene {args.scene!r} "
            f"at frame {frame_idx:06d}: T_map_base must be 4x4"
        )
    replayed_pose = map_from_base_matrix(state)
    if not np.allclose(
        replayed_pose,
        expected_pose,
        rtol=0.0,
        atol=RESUME_POSE_ATOL,
    ):
        max_abs_error = float(
            np.max(np.abs(replayed_pose - expected_pose))
        )
        raise RuntimeError(
            f"Resume replay divergence for scene {args.scene!r} at frame "
            f"{frame_idx:06d}: pose exceeds atol={RESUME_POSE_ATOL:g}; "
            f"max_abs_error={max_abs_error:.9g}"
        )


def _initialize_trajectory(
    sim,
    args: argparse.Namespace,
    existing_manifest: Sequence[Dict[str, object]],
) -> TrajectoryState:
    """Initialize from the seed and deterministically replay committed poses."""
    initialize_agent(sim, args)
    trajectory = TrajectoryState()
    for frame_idx, record in enumerate(existing_manifest):
        if frame_idx > 0:
            transition = _advance_trajectory(
                sim,
                args,
                frame_idx,
                trajectory,
                render_observations=False,
            )
            trajectory = transition.trajectory
        state = sim.get_agent(0).get_state()
        _validate_replayed_frame(args, frame_idx, record, state)
    return trajectory


def generate_scene(
    args: argparse.Namespace,
    scene_files: ReplicaSceneFiles,
    scene_split: str,
    writer: RobotBEVWriter,
) -> Dict[str, object]:
    """Render one validated Replica scene into the canonical writer."""
    _reject_unsupported_visualization_options(args)
    hs = _require_replica_habitat(args, rendering=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.use_physics = bool(
        args.enable_physics and not args.disable_physics
    )

    xbound = parse_bound(args.xbound, "xbound")
    ybound = parse_bound(args.ybound, "ybound")
    existing_manifest = _load_canonical_manifest(writer, args.scene)
    if existing_manifest and not args.resume:
        raise RuntimeError(
            f"Scene {args.scene!r} already contains completed frames; "
            "use --resume or choose a new output directory"
        )

    scene_output_dir = Path(writer.root) / args.scene

    intrinsic = make_camera_intrinsic(args.width, args.height, args.hfov)
    print(f"Dataset: {scene_files.dataset_config}")
    print(f"Scene: {args.scene} ({scene_split})")
    print(f"Output: {scene_output_dir}")
    print(
        f"Replica PTex atlases={scene_files.ptex_atlas_count} "
        f"bytes={sum(path.stat().st_size for path in scene_files.ptex_atlases)}"
    )
    print(
        f"Habitat-Sim={getattr(hs, '__version__', 'unknown')} "
        f"GPU={args.gpu_id}"
    )

    cfg = make_cfg(args)
    with hs.Simulator(cfg) as sim:
        stage_template = sim.get_stage_initialization_template()
        if stage_template is None:
            raise RuntimeError(
                f"No initialized stage template for Replica scene {args.scene!r}"
            )
        render_asset_type = int(stage_template.render_asset_type)
        if render_asset_type != FRL_PTEX_ASSET_TYPE:
            raise RuntimeError(
                "Habitat did not classify the Replica render mesh as "
                "FRL_PTEX_MESH "
                f"(expected asset type {FRL_PTEX_ASSET_TYPE}, "
                f"found {render_asset_type}). Refusing a vertex-color fallback."
            )
        sim.seed(args.seed)
        initialize_navmesh(sim, args)
        trajectory = _initialize_trajectory(
            sim, args, existing_manifest
        )
        if existing_manifest:
            print(
                f"Replayed {len(existing_manifest)} committed frames; "
                f"resuming after frame {len(existing_manifest) - 1:06d} "
                f"collided={trajectory.last_collided} "
                f"stair_recoveries={trajectory.stair_recoveries}"
            )

        current_state = sim.get_agent(0).get_state()
        navmesh_topdown = build_navmesh_topdown(
            sim,
            min(xbound[2], ybound[2]),
            float(current_state.position[1]),
        )
        semantic_id_to_class = build_semantic_id_to_class(sim)
        if not semantic_id_to_class:
            raise RuntimeError(
                "Replica semantic scene loaded zero instance mappings. "
                "Check that the stage config uses info_semantic.json and "
                "mesh_semantic.ply."
            )
        print(f"Semantic id mappings={len(semantic_id_to_class)}")

        for frame_idx in range(len(existing_manifest), args.num_frames):
            if frame_idx == 0:
                obs = sim.get_sensor_observations()
            else:
                transition = _advance_trajectory(
                    sim,
                    args,
                    frame_idx,
                    trajectory,
                    render_observations=True,
                )
                trajectory = transition.trajectory
                obs = transition.observations
                if obs is None:
                    raise RuntimeError(
                        "live trajectory transition returned no observations"
                    )

            state = sim.get_agent(0).get_state()
            timestamp = (
                args.timestamp_start + frame_idx * args.timestamp_step
            )
            rgb = np.asarray(obs[RGB_UUID])[:, :, :3].astype(np.uint8)
            depth = np.asarray(obs[DEPTH_UUID], dtype=np.float32)
            semantic_obs = np.asarray(obs[SEMANTIC_UUID])
            if (
                semantic_obs.size == 0
                or int(np.max(semantic_obs)) > np.iinfo(np.uint16).max
            ):
                raise RuntimeError(
                    "Front semantic observation is empty or exceeds uint16 range"
                )
            depth_mm = np.zeros(depth.shape, dtype=np.uint16)
            depth_valid = np.isfinite(depth) & (depth > 0.0)
            depth_mm[depth_valid] = np.clip(
                depth[depth_valid] * 1000.0, 0.0, 65535.0
            ).astype(np.uint16)

            t_base_camera_habitat = sensor_to_base_matrix(
                state, DEPTH_UUID
            )
            points, front_semantic_ids = depth_to_points(
                depth,
                intrinsic,
                t_base_camera_habitat,
                args.max_depth,
                args.depth_stride,
                args.max_points,
                semantic_obs,
            )
            views: List[
                Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]
            ] = [
                (
                    points,
                    front_semantic_ids,
                    t_base_camera_habitat[:3, 3],
                )
            ]
            valid_mask = make_observation_mask(views, xbound, ybound)
            mask = make_bev_labels(
                sim,
                state,
                views,
                semantic_id_to_class,
                xbound,
                ybound,
                args.min_obstacle_height,
                args.max_obstacle_height,
                valid_mask,
                navmesh_topdown,
            )

            payload = FramePayload(
                frame_id=frame_idx,
                timestamp=timestamp,
                rgb=rgb,
                points=points.astype(np.float32, copy=False),
                bev_labels=mask.astype(np.uint8, copy=False),
                observed_mask=valid_mask.astype(np.uint8, copy=False),
                class_validity=np.ones(
                    (len(MAP_CLASSES),), dtype=np.uint8
                ),
                cam_intrinsic=intrinsic.astype(np.float32, copy=False),
                camera2base=camera_optical_to_base_matrix(
                    t_base_camera_habitat
                ),
                lidar2base=np.eye(4, dtype=np.float32),
                map_from_base=map_from_base_matrix(state),
                depth_mm=depth_mm,
                semantics=semantic_obs.astype(np.uint16, copy=False),
            )
            writer.write_frame(args.scene, scene_split, payload)
            print(
                f"[{frame_idx + 1:06d}/{args.num_frames:06d}] "
                f"points={points.shape[0]} "
                f"valid={float(np.mean(valid_mask)):.3f} "
                f"mask={tuple(mask.shape)}"
            )

        summary = writer.finalize_scene(args.scene, scene_split)
        print(json.dumps(summary, indent=2))
        return summary


def _write_failure_summary(
    writer: RobotBEVWriter,
    summaries: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, str]],
) -> None:
    info_counts = {"train": 0, "val": 0, "test": 0}
    for summary in summaries:
        info_counts[str(summary["split"])] += int(summary["frame_count"])
    payload = {
        "status": "failed",
        "info_counts": info_counts,
        "scene_summaries": list(summaries),
        "failures": list(failures),
    }
    path = Path(writer.root) / "multi_scene_summary.json"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def run_generation(args: argparse.Namespace) -> None:
    """Preflight and generate all requested Replica scenes."""
    _validate_args(args)
    scenes = load_scene_list(args)
    hs = _require_replica_habitat(
        args, rendering=not args.preflight_only
    )
    version = str(getattr(hs, "__version__", "unknown"))

    dataset_config = Path(args.dataset).expanduser().resolve()
    validated = {
        scene: validate_replica_scene(dataset_config, scene)
        for scene in scenes
    }
    assignments = load_scene_splits(
        Path(args.split_file) if args.split_file else None,
        scenes,
    )
    for scene in scenes:
        files = validated[scene]
        print(
            f"Preflight OK: {scene} split={assignments[scene]} "
            f"PTex_atlases={files.ptex_atlas_count} "
            f"Habitat-Sim={version}"
        )
    if args.preflight_only:
        return

    split_lists = {
        split: [
            scene for scene in scenes if assignments[scene] == split
        ]
        for split in ("train", "val", "test")
    }
    writer = RobotBEVWriter(
        root=Path(args.output_dir),
        dataset_id=args.dataset_id,
        source_type="simulation",
        source_dataset="replica_v1",
        generator_name="habitat_replica_robot_bev",
        generator_version="3",
        splits=split_lists,
        generation_parameters=generation_parameters(args),
        resume=args.resume,
    )

    summaries: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for scene_idx, scene in enumerate(scenes):
        scene_args = argparse.Namespace(**vars(args))
        scene_args.scene = scene
        scene_args.seed = args.seed + scene_idx
        scene_args.timestamp_start = (
            args.timestamp_start
            + scene_idx * args.scene_timestamp_stride
        )
        print(f"=== Scene {scene_idx + 1}/{len(scenes)}: {scene} ===")
        try:
            summaries.append(
                generate_scene(
                    scene_args,
                    validated[scene],
                    assignments[scene],
                    writer,
                )
            )
        except Exception as exc:
            failure = {
                "scene": scene,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                f"FAILED scene {scene}: {failure['error']}",
                file=sys.stderr,
            )

    if failures:
        _write_failure_summary(writer, summaries, failures)
        raise RuntimeError(
            f"{len(failures)} Replica scene(s) failed; see "
            f"{Path(writer.root) / 'multi_scene_summary.json'}"
        )
    writer.finalize_dataset()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render original Replica v1 PTex scenes with Habitat-Sim 0.2.2 "
            "and generate RGB, Z-depth, base-frame point clouds, and "
            "six-channel BEV labels."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to original Replica replica.scene_dataset_config.json",
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scene", default="office_1")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--scenes-file")
    parser.add_argument(
        "--split-file", help="JSON with train/val/test scene lists"
    )
    parser.add_argument(
        "--output-dir", default="data/original_replica_robot"
    )
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=120.0)
    parser.add_argument("--zfar", type=float, default=8.0)
    parser.add_argument("--camera-height", type=float, default=1.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--agent-height", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.36)
    parser.add_argument(
        "--agent-max-climb",
        type=float,
        default=0.20,
        help=(
            "Recast mesh-generation tolerance in meters. Runtime stair "
            "safety is controlled separately by --max-floor-height-delta."
        ),
    )
    parser.add_argument(
        "--agent-max-slope",
        type=float,
        default=45.0,
        help=(
            "Recast triangle-slope tolerance in degrees. Runtime stair "
            "safety is controlled separately by --max-floor-height-delta."
        ),
    )
    parser.add_argument(
        "--enable-stair-filter",
        action="store_true",
        help=(
            "Enable the additional local floor-height check for sampled "
            "points and trajectory steps."
        ),
    )
    parser.add_argument("--stair-check-radius", type=float, default=0.50)
    parser.add_argument(
        "--max-floor-height-delta", type=float, default=0.03
    )
    parser.add_argument("--safe-point-max-tries", type=int, default=1000)

    parser.add_argument(
        "--xbound", type=float, nargs=3, default=[0.0, 3.0, 0.02]
    )
    parser.add_argument(
        "--ybound", type=float, nargs=3, default=[-1.5, 1.5, 0.02]
    )
    parser.add_argument(
        "--min-obstacle-height", type=float, default=0.035
    )
    parser.add_argument(
        "--max-obstacle-height", type=float, default=1.05
    )

    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=20000)
    unsupported_visualization_help = (
        "DEPRECATED AND UNSUPPORTED: the canonical writer does not emit "
        "legacy visualization artifacts; this option is rejected"
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help=unsupported_visualization_help,
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help=unsupported_visualization_help,
    )

    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--turn-angle", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timestamp-start", type=int, default=1_000_000)
    parser.add_argument("--timestamp-step", type=int, default=100_000)
    parser.add_argument(
        "--scene-timestamp-stride", type=int, default=10_000_000
    )

    parser.set_defaults(semantic_sensor=True)
    parser.add_argument("--enable-physics", action="store_true")
    parser.add_argument("--disable-physics", action="store_true")
    parser.add_argument(
        "--physics-config", default="data/default.physics_config.json"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--recompute-navmesh", action="store_true")
    parser.add_argument(
        "--navmesh-cell-size",
        type=float,
        default=0.05,
        help="Horizontal Recast voxel size in meters.",
    )
    parser.add_argument(
        "--navmesh-cell-height",
        type=float,
        default=0.20,
        help=(
            "Vertical Recast voxel size in meters. The 0.20 m default "
            "matches stable Habitat-Sim 0.2.2 Replica settings."
        ),
    )
    parser.add_argument(
        "--navmesh-include-static-objects", action="store_true"
    )
    return parser
