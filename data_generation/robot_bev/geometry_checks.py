"""Dependency-light geometry checks for canonical robot BEV datasets."""

import pickle
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .schema import BEV_SHAPE, MAP_CLASSES, MAP_PALETTE, POINT_DIMENSIONS


_BEV_SCALE = 4
_RGB_POINT_COLOR = (255, 64, 64)
_BEV_POINT_COLOR = (255, 255, 255)
_X_AXIS_COLOR = (255, 96, 32)
_Y_AXIS_COLOR = (32, 200, 255)
_CLASS_COLORS = tuple(MAP_PALETTE[name] for name in MAP_CLASSES)
_SWEEP_COLORS = (
    (32, 200, 255),
    (255, 96, 192),
    (255, 176, 32),
    (128, 224, 96),
    (176, 128, 255),
)


def history_to_current_lidar(
    cur_map_from_base: np.ndarray,
    cur_base_from_lidar: np.ndarray,
    hist_map_from_base: np.ndarray,
    hist_base_from_lidar: np.ndarray,
) -> np.ndarray:
    """Return the fixed-contract transform from history to current LiDAR."""
    cur_map_from_lidar = cur_map_from_base @ cur_base_from_lidar
    hist_map_from_lidar = hist_map_from_base @ hist_base_from_lidar
    return np.linalg.inv(cur_map_from_lidar) @ hist_map_from_lidar


def points_to_bev_cells(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map forward/left metric coordinates to canonical BEV rows/columns."""
    rows = np.floor((points[:, 0] - 0.0) / 0.02).astype(np.int64)
    cols = np.floor((points[:, 1] + 1.5) / 0.02).astype(np.int64)
    valid = (rows >= 0) & (rows < 150) & (cols >= 0) & (cols < 150)
    return rows, cols, valid


def project_lidar_to_image(
    points_lidar: np.ndarray,
    camera_from_lidar: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points into an OpenCV optical image."""
    homogeneous = np.concatenate(
        [
            points_lidar[:, :3],
            np.ones((len(points_lidar), 1), dtype=np.float32),
        ],
        axis=1,
    )
    camera = (camera_from_lidar @ homogeneous.T).T[:, :3]
    pixels = (intrinsic @ camera.T).T
    uv = pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-8)
    height, width = image_shape
    valid = (
        (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid


def write_geometry_diagnostics(
    root: Path,
    scene_id: str,
    frame_id: int,
    history_count: int = 5,
) -> Tuple[Path, Path, Path, Path]:
    """Write camera, BEV, and aligned-sweep diagnostics for one frame."""
    if history_count < 0:
        raise ValueError("history_count must be non-negative")

    root = Path(root).expanduser().resolve()
    infos = _load_scene_infos(root, scene_id)
    current_index = next(
        (
            index
            for index, info in enumerate(infos)
            if int(info["frame_id"]) == int(frame_id)
        ),
        None,
    )
    if current_index is None:
        raise ValueError(
            f"scene {scene_id!r} does not contain frame_id {frame_id}"
        )

    current = infos[current_index]
    points_lidar = _load_points(root, current)
    output_dir = root / "diagnostics" / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(frame_id):06d}"
    rgb_path = output_dir / f"{stem}_rgb_point_overlay.png"
    bev_path = output_dir / f"{stem}_bev_overlay.png"
    sweeps_path = output_dir / f"{stem}_aligned_sweeps.png"
    overview_path = output_dir / f"{stem}_overview.png"

    _write_rgb_overlay(root, current, points_lidar, rgb_path)
    _write_bev_overlay(root, current, points_lidar, bev_path)
    history = infos[max(0, current_index - history_count) : current_index]
    _write_sweep_overlay(root, current, points_lidar, history, sweeps_path)
    _write_overview(root, current, points_lidar, overview_path)
    return rgb_path, bev_path, sweeps_path, overview_path


def _load_scene_infos(
    root: Path, scene_id: str
) -> Sequence[Mapping[str, object]]:
    index_path = root / scene_id / "scene_infos.pkl"
    with index_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["infos"]


def _load_points(root: Path, info: Mapping[str, object]) -> np.ndarray:
    values = np.fromfile(root / str(info["lidar_path"]), dtype=np.float32)
    dimension_count = len(POINT_DIMENSIONS)
    if values.size % dimension_count:
        raise ValueError(
            f"point file {info['lidar_path']!r} is not divisible by "
            f"{dimension_count} float32 values"
        )
    return values.reshape(-1, dimension_count)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [
            points[:, :3],
            np.ones((len(points), 1), dtype=np.float32),
        ],
        axis=1,
    )
    return (transform @ homogeneous.T).T[:, :3]


def _write_rgb_overlay(
    root: Path,
    info: Mapping[str, object],
    points_lidar: np.ndarray,
    output_path: Path,
) -> None:
    with Image.open(root / str(info["image_path"])) as source:
        image = source.convert("RGB")
    camera_from_lidar = np.linalg.inv(np.asarray(info["camera2base"])) @ np.asarray(
        info["lidar2base"]
    )
    uv, valid = project_lidar_to_image(
        points_lidar,
        camera_from_lidar,
        np.asarray(info["cam_intrinsic"]),
        (image.height, image.width),
    )
    draw = ImageDraw.Draw(image)
    for u, v in uv[valid]:
        draw.ellipse(
            (float(u) - 2, float(v) - 2, float(u) + 2, float(v) + 2),
            fill=_RGB_POINT_COLOR,
        )
    image.save(output_path, format="PNG")


def _write_bev_overlay(
    root: Path,
    info: Mapping[str, object],
    points_lidar: np.ndarray,
    output_path: Path,
) -> None:
    _render_bev_overlay(root, info, points_lidar).save(output_path, format="PNG")


def _render_bev_overlay(
    root: Path,
    info: Mapping[str, object],
    points_lidar: np.ndarray,
) -> Image.Image:
    labels = np.load(root / str(info["bev_mask_path"]), allow_pickle=False)
    observed = np.load(
        root / str(info["bev_observed_mask_path"]), allow_pickle=False
    )
    image = _bev_image(_bev_label_cells(labels, observed, blended=True))
    draw = ImageDraw.Draw(image)
    points_base = _transform_points(points_lidar, np.asarray(info["lidar2base"]))
    _draw_bev_points(draw, points_base, _BEV_POINT_COLOR)
    with Image.open(root / str(info["image_path"])) as camera_image:
        camera_image_size = camera_image.size
    _draw_camera_frustum(draw, info, camera_image_size)
    _draw_axes(draw, image.size)
    _draw_class_legend(draw)
    return image


def _write_overview(
    root: Path,
    info: Mapping[str, object],
    points_lidar: np.ndarray,
    output_path: Path,
) -> None:
    """Write a six-panel quick-look comparison for one canonical frame."""
    with Image.open(root / str(info["image_path"])) as source:
        rgb = source.convert("RGB")
    labels = np.load(root / str(info["bev_mask_path"]), allow_pickle=False)
    observed = np.load(
        root / str(info["bev_observed_mask_path"]), allow_pickle=False
    )
    panels = (
        ("RGB", rgb),
        ("Depth", _load_optional_depth(root, info, rgb.size)),
        ("Semantic", _load_optional_semantic(root, info, rgb.size)),
        ("BEV labels", _bev_image(_bev_label_cells(labels, observed, blended=False))),
        ("Observed mask", _bev_image(_observed_cells(observed))),
        ("BEV + points", _render_bev_overlay(root, info, points_lidar)),
    )
    _compose_overview(panels, output_path)


def _load_optional_depth(
    root: Path,
    info: Mapping[str, object],
    fallback_size: Tuple[int, int],
) -> Image.Image:
    depth_path = info.get("depth_path")
    if not depth_path:
        return _placeholder(fallback_size, "no depth")
    with Image.open(root / str(depth_path)) as source:
        depth = np.asarray(source)
    return Image.fromarray(_colorize_depth(depth), mode="RGB")


def _load_optional_semantic(
    root: Path,
    info: Mapping[str, object],
    fallback_size: Tuple[int, int],
) -> Image.Image:
    semantic_path = info.get("semantic_path")
    if not semantic_path:
        return _placeholder(fallback_size, "no semantic")
    with Image.open(root / str(semantic_path)) as source:
        semantic = np.asarray(source)
    return Image.fromarray(_colorize_semantic(semantic), mode="RGB")


def _placeholder(size: Tuple[int, int], text: str) -> Image.Image:
    image = Image.new("RGB", size, (24, 24, 28))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill=(220, 220, 220))
    return image


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = values > 0
    colored = np.zeros(values.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return colored
    lo = float(np.percentile(values[valid], 2))
    hi = float(np.percentile(values[valid], 98))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    colored[..., 0] = (255.0 * normalized).astype(np.uint8)
    colored[..., 1] = (255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(
        np.uint8
    )
    colored[..., 2] = (255.0 * (1.0 - normalized)).astype(np.uint8)
    colored[~valid] = (0, 0, 0)
    return colored


def _colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    values = np.asarray(semantic, dtype=np.uint32)
    colored = np.zeros(values.shape + (3,), dtype=np.uint8)
    nonzero = values != 0
    colored[..., 0] = ((values * 37 + 23) % 255).astype(np.uint8)
    colored[..., 1] = ((values * 67 + 71) % 255).astype(np.uint8)
    colored[..., 2] = ((values * 97 + 149) % 255).astype(np.uint8)
    colored[~nonzero] = (0, 0, 0)
    return colored


def _bev_label_cells(
    labels: np.ndarray,
    observed: np.ndarray,
    blended: bool,
) -> np.ndarray:
    cells = np.full(BEV_SHAPE[1:] + (3,), (16, 16, 20), dtype=np.uint8)
    cells[observed.astype(bool)] = (48, 48, 54)
    for class_index, color in enumerate(_CLASS_COLORS):
        mask = labels[class_index].astype(bool)
        if blended:
            cells[mask] = (
                (
                    cells[mask].astype(np.uint16)
                    + np.asarray(color, dtype=np.uint16)
                )
                // 2
            ).astype(np.uint8)
        else:
            cells[mask] = color
    return cells


def _observed_cells(observed: np.ndarray) -> np.ndarray:
    cells = np.zeros(BEV_SHAPE[1:] + (3,), dtype=np.uint8)
    cells[observed.astype(bool)] = (230, 230, 230)
    return cells


def _compose_overview(
    panels: Iterable[Tuple[str, Image.Image]],
    output_path: Path,
) -> None:
    panel_width, panel_height = 320, 240
    title_height = 22
    columns = 3
    rows = 2
    canvas = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + title_height)),
        (10, 10, 12),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = col * panel_width
        y = row * (panel_height + title_height)
        draw.rectangle(
            (x, y, x + panel_width, y + title_height), fill=(32, 32, 36)
        )
        draw.text((x + 8, y + 5), title, fill=(235, 235, 235))
        resized = _fit_panel(image, (panel_width, panel_height))
        canvas.paste(resized, (x, y + title_height))
    canvas.save(output_path, format="PNG")


def _fit_panel(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    source = image.convert("RGB")
    scale = min(target_width / source.width, target_height / source.height)
    width = max(1, int(round(source.width * scale)))
    height = max(1, int(round(source.height * scale)))
    resized = source.resize((width, height), resample=Image.NEAREST)
    panel = Image.new("RGB", size, (0, 0, 0))
    panel.paste(resized, ((target_width - width) // 2, (target_height - height) // 2))
    return panel


def _write_sweep_overlay(
    root: Path,
    current: Mapping[str, object],
    current_points: np.ndarray,
    history: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    cells = np.full(BEV_SHAPE[1:] + (3,), (12, 12, 16), dtype=np.uint8)
    image = _bev_image(cells)
    draw = ImageDraw.Draw(image)
    current_map_from_base = np.asarray(current["T_map_base"])
    current_base_from_lidar = np.asarray(current["lidar2base"])

    for age, history_info in enumerate(reversed(history), start=1):
        history_points = _load_points(root, history_info)
        current_lidar_from_history = history_to_current_lidar(
            current_map_from_base,
            current_base_from_lidar,
            np.asarray(history_info["T_map_base"]),
            np.asarray(history_info["lidar2base"]),
        )
        aligned_lidar = _transform_points(
            history_points, current_lidar_from_history
        )
        aligned_base = _transform_points(aligned_lidar, current_base_from_lidar)
        color = _SWEEP_COLORS[(age - 1) % len(_SWEEP_COLORS)]
        _draw_bev_points(draw, aligned_base, color)
        draw.text((8, 8 + age * 12), f"history -{age}", fill=color)

    current_base = _transform_points(current_points, current_base_from_lidar)
    _draw_bev_points(draw, current_base, _BEV_POINT_COLOR)
    draw.text((8, 8), "current", fill=_BEV_POINT_COLOR)
    _draw_axes(draw, image.size)
    image.save(output_path, format="PNG")


def _bev_image(cells: np.ndarray) -> Image.Image:
    oriented = np.ascontiguousarray(cells[::-1, ::-1])
    return Image.fromarray(oriented, mode="RGB").resize(
        (oriented.shape[1] * _BEV_SCALE, oriented.shape[0] * _BEV_SCALE),
        resample=Image.NEAREST,
    )


def _draw_bev_points(
    draw: ImageDraw.ImageDraw,
    points_base: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    rows, cols, valid = points_to_bev_cells(points_base)
    grid_height, grid_width = BEV_SHAPE[1:]
    for row, col in zip(rows[valid], cols[valid]):
        x = (grid_width - 1 - int(col)) * _BEV_SCALE + _BEV_SCALE // 2
        y = (grid_height - 1 - int(row)) * _BEV_SCALE + _BEV_SCALE // 2
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _draw_camera_frustum(
    draw: ImageDraw.ImageDraw,
    info: Mapping[str, object],
    camera_image_size: Tuple[int, int],
) -> None:
    intrinsic = np.asarray(info["cam_intrinsic"])
    camera2base = np.asarray(info["camera2base"])
    width, height = camera_image_size
    corners = np.array(
        [[0.0, height - 1.0, 1.0], [width - 1.0, height - 1.0, 1.0]]
    )
    rays_camera = (np.linalg.inv(intrinsic) @ corners.T).T
    rays_base = (camera2base[:3, :3] @ rays_camera.T).T
    origin = camera2base[:3, 3]
    footprint = []
    for ray in rays_base:
        if abs(float(ray[2])) > 1e-8:
            ground_distance = -float(origin[2]) / float(ray[2])
        else:
            ground_distance = None
        distance = _ray_distance_inside_bev(origin, ray, ground_distance)
        if distance is None:
            continue
        footprint.append(origin + distance * ray)

    grid_height, grid_width = BEV_SHAPE[1:]

    def pixel(point):
        rows, cols, valid = points_to_bev_cells(np.asarray([point]))
        if not valid[0]:
            return None
        return (
            (grid_width - 1 - int(cols[0])) * _BEV_SCALE + _BEV_SCALE // 2,
            (grid_height - 1 - int(rows[0])) * _BEV_SCALE + _BEV_SCALE // 2,
        )

    origin_pixel = pixel(origin)
    if origin_pixel is None:
        return
    for point in footprint:
        endpoint = pixel(point)
        if endpoint is not None:
            draw.line((origin_pixel, endpoint), fill=(255, 220, 64), width=2)


def _ray_distance_inside_bev(
    origin: np.ndarray,
    ray: np.ndarray,
    ground_distance: Optional[float],
) -> Optional[float]:
    boundary_distances = []
    for coordinate, direction, lower, upper in (
        (float(origin[0]), float(ray[0]), 0.0, 3.0 - 1e-6),
        (float(origin[1]), float(ray[1]), -1.5, 1.5 - 1e-6),
    ):
        if direction > 1e-8:
            boundary_distances.append((upper - coordinate) / direction)
        elif direction < -1e-8:
            boundary_distances.append((lower - coordinate) / direction)
    positive_boundaries = [value for value in boundary_distances if value > 0]
    if not positive_boundaries:
        return None
    boundary_distance = min(positive_boundaries)
    if ground_distance is not None and ground_distance > 0:
        return min(ground_distance, boundary_distance)
    return boundary_distance


def _draw_axes(draw: ImageDraw.ImageDraw, image_size: Tuple[int, int]) -> None:
    width, height = image_size
    origin = (width - 70, height - 50)
    x_target = (origin[0], origin[1] - 85)
    y_target = (origin[0] - 95, origin[1])
    draw.line((origin, x_target), fill=_X_AXIS_COLOR, width=4)
    draw.line(
        (x_target, (x_target[0] - 6, x_target[1] + 10)),
        fill=_X_AXIS_COLOR,
        width=4,
    )
    draw.line(
        (x_target, (x_target[0] + 6, x_target[1] + 10)),
        fill=_X_AXIS_COLOR,
        width=4,
    )
    draw.text((x_target[0] + 6, x_target[1]), "x forward", fill=_X_AXIS_COLOR)
    draw.line((origin, y_target), fill=_Y_AXIS_COLOR, width=4)
    draw.line(
        (y_target, (y_target[0] + 10, y_target[1] - 6)),
        fill=_Y_AXIS_COLOR,
        width=4,
    )
    draw.line(
        (y_target, (y_target[0] + 10, y_target[1] + 6)),
        fill=_Y_AXIS_COLOR,
        width=4,
    )
    draw.text((y_target[0], y_target[1] + 8), "y left", fill=_Y_AXIS_COLOR)


def _draw_class_legend(draw: ImageDraw.ImageDraw) -> None:
    for index, (name, color) in enumerate(zip(MAP_CLASSES, _CLASS_COLORS)):
        y = 8 + index * 12
        draw.rectangle((8, y, 15, y + 7), fill=color)
        draw.text((20, y - 2), name, fill=color)
