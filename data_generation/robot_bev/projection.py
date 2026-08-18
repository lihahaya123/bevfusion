"""Source-independent depth projection and BEV rasterization helpers."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple

import numpy as np

from .schema import MAP_CLASSES


Bound3 = Tuple[float, float, float]
Bound2 = Tuple[float, float]


def depth_to_base_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    camera2base: np.ndarray,
    *,
    pixel_mask: Optional[np.ndarray] = None,
    stride: int = 1,
    max_depth: float = float("inf"),
    max_points: Optional[int] = None,
    semantic: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Back-project OpenCV Z-depth into x-forward/y-left/z-up base points."""
    depth = np.asarray(depth_m, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    camera2base = np.asarray(camera2base, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be two-dimensional")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must be a 3x3 matrix")
    if camera2base.shape != (4, 4):
        raise ValueError("camera2base must be a 4x4 matrix")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive when provided")

    rows = np.arange(0, depth.shape[0], stride, dtype=np.int64)
    cols = np.arange(0, depth.shape[1], stride, dtype=np.int64)
    uu, vv = np.meshgrid(cols, rows)
    dd = depth[vv, uu]
    valid = np.isfinite(dd) & (dd > 0.0) & (dd < max_depth)
    if pixel_mask is not None:
        mask = np.asarray(pixel_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("pixel_mask must match depth shape")
        valid &= mask[vv, uu]
    if not np.any(valid):
        empty = np.zeros((0, 5), dtype=np.float32)
        return empty, None if semantic is None else np.zeros((0,), dtype=np.int64)

    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    z = dd[valid].astype(np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsic focal lengths must be positive")

    points_camera = np.stack(
        [
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z,
        ],
        axis=1,
    ).astype(np.float32)
    points_base = (
        points_camera @ camera2base[:3, :3].T + camera2base[:3, 3]
    )

    semantic_ids = None
    if semantic is not None:
        semantic_array = np.asarray(semantic)
        if semantic_array.shape != depth.shape:
            raise ValueError("semantic must match depth shape")
        semantic_ids = semantic_array[vv[valid], uu[valid]].astype(np.int64)

    if max_points is not None and points_base.shape[0] > max_points:
        keep = np.linspace(
            0, points_base.shape[0] - 1, max_points, dtype=np.int64
        )
        points_base = points_base[keep]
        if semantic_ids is not None:
            semantic_ids = semantic_ids[keep]

    attributes = np.zeros((points_base.shape[0], 2), dtype=np.float32)
    points = np.concatenate(
        [points_base.astype(np.float32, copy=False), attributes], axis=1
    )
    return points, semantic_ids


def point_indices(
    points: np.ndarray,
    xbound: Bound3,
    ybound: Bound3,
    zbound: Optional[Bound2] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return BEV rows, columns, and the corresponding source-point mask."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N,>=3]")
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    valid = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )
    if zbound is not None:
        valid &= (points[:, 2] > zbound[0]) & (points[:, 2] < zbound[1])
    rows = np.floor((points[valid, 0] - x_min) / x_step).astype(np.int64)
    cols = np.floor((points[valid, 1] - y_min) / y_step).astype(np.int64)
    height = int(round((x_max - x_min) / x_step))
    width = int(round((y_max - y_min) / y_step))
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    return rows, cols, valid


def rasterize_semantics(
    points: np.ndarray,
    semantic_ids: np.ndarray,
    semantic_id_to_class: Mapping[int, str],
    xbound: Bound3,
    ybound: Bound3,
    zbound: Bound2,
) -> np.ndarray:
    """Rasterize semantic points into canonical multi-hot BEV channels."""
    points = np.asarray(points)
    semantic_ids = np.asarray(semantic_ids)
    if semantic_ids.shape != (points.shape[0],):
        raise ValueError("semantic_ids must contain one value per point")
    height = int(round((xbound[1] - xbound[0]) / xbound[2]))
    width = int(round((ybound[1] - ybound[0]) / ybound[2]))
    labels = np.zeros((len(MAP_CLASSES), height, width), dtype=np.uint8)
    for semantic_id, class_name in semantic_id_to_class.items():
        if class_name not in MAP_CLASSES:
            raise ValueError(f"unknown canonical class: {class_name!r}")
        selected = points[semantic_ids == int(semantic_id)]
        if selected.size == 0:
            continue
        rows, cols, _ = point_indices(selected, xbound, ybound, zbound)
        labels[MAP_CLASSES.index(class_name), rows, cols] = 1
    return labels


def mark_observed_rays(
    observed_mask: np.ndarray,
    points: np.ndarray,
    camera_origin: np.ndarray,
    xbound: Bound3,
    ybound: Bound3,
    *,
    angular_resolution_deg: float = 0.5,
) -> None:
    """Mark BEV cells traversed by the farthest point in each azimuth bin."""
    if angular_resolution_deg <= 0.0:
        raise ValueError("angular_resolution_deg must be positive")
    points = np.asarray(points)
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
    bins = np.round(
        angles / math.radians(angular_resolution_deg)
    ).astype(np.int32)
    sample_step = min(xbound[2], ybound[2]) * 0.5
    for angle_bin in np.unique(bins):
        in_bin = bins == angle_bin
        farthest = int(np.argmax(np.where(in_bin, ranges, -1.0)))
        end = origin + delta[farthest]
        distance = float(ranges[farthest])
        count = max(2, int(math.ceil(distance / sample_step)) + 1)
        alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
        samples = origin[None, :] + alpha[:, None] * (end - origin)[None, :]
        inside = (
            (samples[:, 0] >= xbound[0])
            & (samples[:, 0] < xbound[1])
            & (samples[:, 1] >= ybound[0])
            & (samples[:, 1] < ybound[1])
        )
        rows = np.floor(
            (samples[inside, 0] - xbound[0]) / xbound[2]
        ).astype(np.int64)
        cols = np.floor(
            (samples[inside, 1] - ybound[0]) / ybound[2]
        ).astype(np.int64)
        observed_mask[rows, cols] = 1


def make_observation_mask(
    points: np.ndarray,
    camera_origin: np.ndarray,
    xbound: Bound3,
    ybound: Bound3,
    *,
    angular_resolution_deg: float = 0.5,
) -> np.ndarray:
    """Build a binary BEV mask from trusted depth rays."""
    height = int(round((xbound[1] - xbound[0]) / xbound[2]))
    width = int(round((ybound[1] - ybound[0]) / ybound[2]))
    observed = np.zeros((height, width), dtype=np.uint8)
    mark_observed_rays(
        observed,
        points,
        camera_origin,
        xbound,
        ybound,
        angular_resolution_deg=angular_resolution_deg,
    )
    return observed


def rasterize_point_mask(
    points: np.ndarray,
    xbound: Bound3,
    ybound: Bound3,
    zbound: Bound2,
) -> np.ndarray:
    """Rasterize point endpoints into a binary BEV mask."""
    height = int(round((xbound[1] - xbound[0]) / xbound[2]))
    width = int(round((ybound[1] - ybound[0]) / ybound[2]))
    mask = np.zeros((height, width), dtype=np.uint8)
    rows, cols, _ = point_indices(points, xbound, ybound, zbound)
    mask[rows, cols] = 1
    return mask


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a two-dimensional binary mask without optional dependencies."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant")
    output = np.zeros_like(mask)
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            output = np.maximum(
                output,
                padded[
                    row_offset : row_offset + mask.shape[0],
                    col_offset : col_offset + mask.shape[1],
                ],
            )
    return output


def semantic_boundary_mask(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """Return pixels near a categorical transition, including both sides."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    boundary = np.zeros(labels.shape, dtype=np.uint8)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:][horizontal] = 1
    boundary[:, :-1][horizontal] = 1
    boundary[1:, :][vertical] = 1
    boundary[:-1, :][vertical] = 1
    return dilate_binary(boundary, radius)
