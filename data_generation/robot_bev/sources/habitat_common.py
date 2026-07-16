"""Habitat-Sim helpers shared by robot BEV source adapters."""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import magnum as mn
    import habitat_sim
    from habitat_sim.utils.common import (
        quat_from_angle_axis,
        quat_from_coeffs,
        quat_to_coeffs,
    )
except ImportError:
    mn = None
    habitat_sim = None
    quat_from_angle_axis = None
    quat_from_coeffs = None
    quat_to_coeffs = None


RGB_UUID = "front_rgb"
DEPTH_UUID = "front_depth"
SEMANTIC_UUID = "front_semantic"


def require_habitat_sim():
    if habitat_sim is None:
        raise RuntimeError(
            "Habitat-Sim is required for rendering; activate the habitat022 environment"
        )
    return habitat_sim


def make_camera_intrinsic(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def camera_to_base_matrix(
    camera_height: float, camera_pitch_deg: float
) -> np.ndarray:
    # p_base[x_forward, y_left, z_up] = R * p_camera[x_right, y_up, z_back] + t
    habitat_to_base = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sensor_pitch = math.radians(camera_pitch_deg)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = habitat_to_base @ rotation_x(sensor_pitch)
    out[:3, 3] = np.array([0.0, 0.0, camera_height], dtype=np.float32)
    return out


def camera_optical_to_base_matrix(
    t_base_camera_habitat: np.ndarray,
) -> np.ndarray:
    """Convert a Habitat/OpenGL camera extrinsic to OpenCV optical axes."""
    optical_to_habitat = np.eye(4, dtype=np.float32)
    optical_to_habitat[1, 1] = -1.0
    optical_to_habitat[2, 2] = -1.0
    return (
        np.asarray(t_base_camera_habitat, dtype=np.float32)
        @ optical_to_habitat
    )


def quat_to_rotation_matrix(rotation) -> np.ndarray:
    require_habitat_sim()
    coeffs = quat_to_coeffs(rotation)
    x, y, z, w = [float(v) for v in coeffs]
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


def map_from_base_matrix(state) -> np.ndarray:
    # Base frame is [forward, left, up]. Habitat local frame is [right, up, back].
    robot_to_habitat = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = quat_to_rotation_matrix(state.rotation) @ robot_to_habitat
    out[:3, 3] = np.asarray(state.position, dtype=np.float32)
    return out


def map_from_habitat_pose(position: Sequence[float], rotation) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = quat_to_rotation_matrix(rotation)
    out[:3, 3] = np.asarray(position, dtype=np.float32)
    return out


def sensor_to_base_matrix(state, sensor_uuid: str) -> np.ndarray:
    """Derive the exact sensor extrinsic returned by Habitat, including all DOF."""
    require_habitat_sim()
    if sensor_uuid not in state.sensor_states:
        raise KeyError(f"Agent state has no sensor named {sensor_uuid!r}")
    sensor_state = state.sensor_states[sensor_uuid]
    t_map_base = map_from_base_matrix(state)
    t_map_sensor = map_from_habitat_pose(
        sensor_state.position, sensor_state.rotation
    )
    return np.linalg.inv(t_map_base) @ t_map_sensor


def base_grid_to_habitat_local(
    x_forward: np.ndarray, y_left: np.ndarray
) -> np.ndarray:
    pts = np.zeros((x_forward.size, 3), dtype=np.float32)
    pts[:, 0] = -y_left.reshape(-1)
    pts[:, 2] = -x_forward.reshape(-1)
    return pts


def transform_habitat_local_to_world(state, local: np.ndarray) -> np.ndarray:
    rot = quat_to_rotation_matrix(state.rotation)
    return local @ rot.T + np.asarray(state.position, dtype=np.float32)


def make_cfg(args: argparse.Namespace):
    hs = require_habitat_sim()
    sim_cfg = hs.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = args.dataset
    sim_cfg.scene_id = args.scene
    sim_cfg.enable_physics = args.use_physics
    sim_cfg.physics_config_file = args.physics_config
    sim_cfg.gpu_device_id = args.gpu_id
    sim_cfg.frustum_culling = True

    sensor_specs = []
    sensor_orientation = mn.Vector3(
        math.radians(args.camera_pitch_deg), 0.0, 0.0
    )

    rgb_spec = hs.CameraSensorSpec()
    rgb_spec.uuid = RGB_UUID
    rgb_spec.sensor_type = hs.SensorType.COLOR
    rgb_spec.sensor_subtype = hs.SensorSubType.PINHOLE
    rgb_spec.resolution = mn.Vector2i([args.height, args.width])
    rgb_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
    rgb_spec.orientation = sensor_orientation
    rgb_spec.hfov = mn.Deg(args.hfov)
    rgb_spec.far = args.zfar
    sensor_specs.append(rgb_spec)

    depth_spec = hs.CameraSensorSpec()
    depth_spec.uuid = DEPTH_UUID
    depth_spec.sensor_type = hs.SensorType.DEPTH
    depth_spec.sensor_subtype = hs.SensorSubType.PINHOLE
    depth_spec.channels = 1
    depth_spec.resolution = mn.Vector2i([args.height, args.width])
    depth_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
    depth_spec.orientation = sensor_orientation
    depth_spec.hfov = mn.Deg(args.hfov)
    depth_spec.far = args.zfar
    sensor_specs.append(depth_spec)

    if args.semantic_sensor:
        semantic_spec = hs.CameraSensorSpec()
        semantic_spec.uuid = SEMANTIC_UUID
        semantic_spec.sensor_type = hs.SensorType.SEMANTIC
        semantic_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        semantic_spec.channels = 1
        semantic_spec.resolution = mn.Vector2i([args.height, args.width])
        semantic_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
        semantic_spec.orientation = sensor_orientation
        semantic_spec.hfov = mn.Deg(args.hfov)
        semantic_spec.far = args.zfar
        sensor_specs.append(semantic_spec)

    agent_cfg = hs.AgentConfiguration()
    agent_cfg.height = args.agent_height
    agent_cfg.radius = args.agent_radius
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": hs.agent.ActionSpec(
            "move_forward", hs.agent.ActuationSpec(amount=args.step_size)
        ),
        "turn_left": hs.agent.ActionSpec(
            "turn_left", hs.agent.ActuationSpec(amount=args.turn_angle)
        ),
        "turn_right": hs.agent.ActionSpec(
            "turn_right", hs.agent.ActuationSpec(amount=args.turn_angle)
        ),
    }

    return hs.Configuration(sim_cfg, [agent_cfg])


def configure_navmesh_settings(
    nav_settings, args: argparse.Namespace
) -> None:
    """Apply settings supported by both Habitat-Sim 0.2.2 and newer releases."""
    require_habitat_sim()
    nav_settings.set_defaults()
    nav_settings.cell_size = args.navmesh_cell_size
    nav_settings.cell_height = args.navmesh_cell_height
    nav_settings.agent_height = args.agent_height
    nav_settings.agent_radius = args.agent_radius
    nav_settings.agent_max_climb = args.agent_max_climb
    nav_settings.agent_max_slope = args.agent_max_slope
    nav_settings.filter_ledge_spans = True
    nav_settings.filter_walkable_low_height_spans = True
    if hasattr(nav_settings, "include_static_objects"):
        nav_settings.include_static_objects = (
            args.navmesh_include_static_objects
        )
    elif args.navmesh_include_static_objects:
        raise RuntimeError(
            "--navmesh-include-static-objects is unavailable in Habitat-Sim 0.2.2. "
            "Original Replica's static stage mesh is included automatically; omit this flag."
        )


def initialize_navmesh(sim, args: argparse.Namespace) -> None:
    hs = require_habitat_sim()
    nav_area = sim.pathfinder.navigable_area if sim.pathfinder.is_loaded else 0.0
    if sim.pathfinder.is_loaded and nav_area > 0.0 and not args.recompute_navmesh:
        print(f"Using dataset navmesh: area={nav_area:.3f}")
        print(
            "WARNING: --agent-max-climb and --agent-max-slope only affect recomputed "
            "navmeshes. Runtime stair filtering is still active."
        )
        return

    print("Recomputing navmesh.")
    nav_settings = hs.NavMeshSettings()
    configure_navmesh_settings(nav_settings, args)
    if not sim.recompute_navmesh(sim.pathfinder, nav_settings):
        raise RuntimeError("Failed to load or recompute navmesh.")
    print(
        f"Navmesh area={sim.pathfinder.navigable_area:.3f} "
        f"cell={args.navmesh_cell_size:.3f}x{args.navmesh_cell_height:.3f} "
        f"max_climb={args.agent_max_climb:.3f} "
        f"max_slope={args.agent_max_slope:.1f}"
    )


def is_floor_level_safe(
    sim,
    position: Sequence[float],
    radius: float,
    max_height_delta: float,
) -> bool:
    require_habitat_sim()
    center = np.asarray(position, dtype=np.float32)
    if not sim.pathfinder.is_navigable(
        center, max_y_delta=max(0.5, max_height_delta)
    ):
        return False
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
            [radius, 0.0, radius],
            [radius, 0.0, -radius],
            [-radius, 0.0, radius],
            [-radius, 0.0, -radius],
        ],
        dtype=np.float32,
    )

    snapped_heights = []
    max_horizontal_snap = max(0.10, radius * 0.5)
    for offset in offsets:
        sample = center + offset
        snapped = np.asarray(sim.pathfinder.snap_point(sample), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            continue
        horizontal_snap = float(
            np.linalg.norm(snapped[[0, 2]] - sample[[0, 2]])
        )
        if horizontal_snap > max_horizontal_snap:
            continue
        snapped_heights.append(float(snapped[1]))
    if not snapped_heights:
        return False
    return max(snapped_heights) - min(snapped_heights) <= max_height_delta


def sample_safe_navigable_point(sim, args: argparse.Namespace) -> np.ndarray:
    require_habitat_sim()
    for _ in range(args.safe_point_max_tries):
        point = np.asarray(
            sim.pathfinder.get_random_navigable_point(), dtype=np.float32
        )
        if not np.all(np.isfinite(point)):
            continue
        if not args.enable_stair_filter or is_floor_level_safe(
            sim,
            point,
            args.stair_check_radius,
            args.max_floor_height_delta,
        ):
            return point
    if args.enable_stair_filter:
        raise RuntimeError(
            "Failed to sample a stair-safe navigable point. Disable "
            "--enable-stair-filter or relax --max-floor-height-delta."
        )
    raise RuntimeError(
        "Failed to sample a finite navigable point from the navmesh."
    )


def initialize_agent(sim, args: argparse.Namespace) -> None:
    hs = require_habitat_sim()
    state = hs.AgentState()
    state.position = sample_safe_navigable_point(sim, args)
    yaw = random.uniform(-math.pi, math.pi)
    state.rotation = quat_from_angle_axis(
        yaw, np.array([0.0, 1.0, 0.0])
    )
    sim.initialize_agent(0, state)


def depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    t_base_camera_habitat: np.ndarray,
    max_depth: float,
    stride: int,
    max_points: int,
    semantic: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    depth = np.asarray(depth, dtype=np.float32)
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    uu, vv = np.meshgrid(cols, rows)
    dd = depth[vv, uu]
    valid = (dd > 0.0) & np.isfinite(dd) & (dd < max_depth)
    if not np.any(valid):
        return np.zeros((0, 5), dtype=np.float32), None

    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    d = dd[valid].astype(np.float32)
    semantic_ids = None
    if semantic is not None:
        semantic_arr = np.asarray(semantic)
        semantic_ids = semantic_arr[vv[valid], uu[valid]].astype(np.int64)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_right = (u - cx) * d / fx
    y_up = -(v - cy) * d / fy
    z_back = -d

    pts_camera = np.stack([x_right, y_up, z_back], axis=1).astype(
        np.float32
    )
    t_base_camera = np.asarray(t_base_camera_habitat, dtype=np.float32)
    if t_base_camera.shape != (4, 4):
        raise ValueError("t_base_camera_habitat must be a 4x4 matrix")
    pts = (
        pts_camera @ t_base_camera[:3, :3].T
        + t_base_camera[:3, 3]
    )
    if pts.shape[0] > max_points:
        keep = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[keep]
        if semantic_ids is not None:
            semantic_ids = semantic_ids[keep]

    intensity_time = np.zeros((pts.shape[0], 2), dtype=np.float32)
    points = np.concatenate(
        [pts.astype(np.float32), intensity_time], axis=1
    )
    return points, semantic_ids


def mark_observed_rays(
    valid_mask: np.ndarray,
    points: np.ndarray,
    camera_origin: np.ndarray,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    angular_resolution_deg: float = 0.5,
) -> None:
    """Mark BEV cells traversed by observed depth rays."""
    if points.size == 0:
        return
    origin = np.asarray(camera_origin, dtype=np.float32)[:2]
    delta = np.asarray(points[:, :2], dtype=np.float32) - origin[None, :]
    ranges = np.linalg.norm(delta, axis=1)
    finite = np.isfinite(ranges) & (ranges > 1e-4)
    if not np.any(finite):
        return
    delta = delta[finite]
    ranges = ranges[finite]
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    bin_width = math.radians(angular_resolution_deg)
    bins = np.round(angles / bin_width).astype(np.int32)

    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    sample_step = min(x_step, y_step) * 0.5
    for angle_bin in np.unique(bins):
        in_bin = bins == angle_bin
        farthest = int(np.argmax(np.where(in_bin, ranges, -1.0)))
        end = origin + delta[farthest]
        distance = float(ranges[farthest])
        count = max(2, int(math.ceil(distance / sample_step)) + 1)
        alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
        samples = (
            origin[None, :]
            + alpha[:, None] * (end - origin)[None, :]
        )
        inside = (
            (samples[:, 0] >= x_min)
            & (samples[:, 0] < x_max)
            & (samples[:, 1] >= y_min)
            & (samples[:, 1] < y_max)
        )
        if not np.any(inside):
            continue
        rows = np.floor(
            (samples[inside, 0] - x_min) / x_step
        ).astype(np.int64)
        cols = np.floor(
            (samples[inside, 1] - y_min) / y_step
        ).astype(np.int64)
        rows = np.clip(rows, 0, valid_mask.shape[0] - 1)
        cols = np.clip(cols, 0, valid_mask.shape[1] - 1)
        valid_mask[rows, cols] = 1


def make_observation_mask(
    views: Sequence[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
) -> np.ndarray:
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    height = int(round((x_max - x_min) / x_step))
    width = int(round((y_max - y_min) / y_step))
    valid_mask = np.zeros((height, width), dtype=np.uint8)
    for points, _, sensor_origin in views:
        mark_observed_rays(
            valid_mask,
            points,
            sensor_origin,
            xbound,
            ybound,
        )
    return valid_mask


def next_action(last_collided: bool, rng: random.Random) -> str:
    if last_collided:
        return rng.choice(["turn_left", "turn_right"])
    return rng.choices(
        ["move_forward", "turn_left", "turn_right"],
        weights=[0.75, 0.125, 0.125],
        k=1,
    )[0]


def turn_agent_away(sim, previous_state, rng: random.Random) -> None:
    hs = require_habitat_sim()
    state = hs.AgentState()
    state.position = np.asarray(previous_state.position, dtype=np.float32)
    turn = rng.choice([-math.pi / 2.0, math.pi / 2.0, math.pi])
    state.rotation = previous_state.rotation * quat_from_angle_axis(
        turn,
        np.array([0.0, 1.0, 0.0]),
    )
    sim.get_agent(0).set_state(state)


def state_from_manifest(record: Dict[str, object]):
    """Restore a Habitat agent pose from a canonical writer manifest record."""
    hs = require_habitat_sim()
    transform = np.asarray(record["T_map_base"], dtype=np.float32)
    if transform.shape != (4, 4):
        raise RuntimeError("Canonical manifest T_map_base must be a 4x4 matrix")
    robot_to_habitat = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    habitat_rotation = transform[:3, :3] @ robot_to_habitat.T
    quaternion = mn.Quaternion.from_matrix(mn.Matrix3(habitat_rotation))
    coefficients = np.array(
        [*quaternion.vector, quaternion.scalar], dtype=np.float64
    )
    state = hs.AgentState()
    state.position = transform[:3, 3]
    state.rotation = quat_from_coeffs(coefficients)
    return state
