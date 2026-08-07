"""Convert self-collected Depth/Left/in.txt data into prepare_mytest_robot_bev.py input.

The source layout is:

    <src-root>/Depth/*.png     uint16 depth maps, values in mm
    <src-root>/Left/*.jpg      left RGB frames, timestamp prefix matches Depth
    <src-root>/in.txt          fx fy cx cy

The converter writes the canonical mytest source layout:

    <dst-root>/rgb/<timestamp>.png
    <dst-root>/pclCam/pointcloud_depth_<timestamp>_LOS_<timestamp>_0.txt
    <dst-root>/in.txt

Depth points are unprojected with the camera intrinsics and converted from
camera optical coordinates (x right, y down, z forward) to the robot/ego
convention used by the existing mytest data (x forward, y left, z up), keeping
millimetre units so prepare_mytest_robot_bev.py's default --point-scale 0.001
produces metre point clouds.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# p_ego = CAMERA_TO_EGO @ p_camera
CAMERA_TO_EGO = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare latest self-collected Depth/Left data for "
            "prepare_mytest_robot_bev.py."
        )
    )
    parser.add_argument(
        "--src-root",
        default="E:/lxx/V4/selfcollect/data20260807",
        type=Path,
        help="Input directory containing Depth/, Left/ and in.txt.",
    )
    parser.add_argument(
        "--dst-root",
        default="data/mytest/data_20260807",
        type=Path,
        help="Output mytest source directory.",
    )
    parser.add_argument(
        "--robot-bev-out",
        default="data/mytest/robot_bev_20260807",
        type=Path,
        help="Output root passed to prepare_mytest_robot_bev.py.",
    )
    parser.add_argument("--scene-id", default="mytest_20260807")
    parser.add_argument("--dataset-id", default="mytest_robot_bev_v4_20260807")
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help="History sweeps passed to prepare_mytest_robot_bev.py.",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=8,
        help="Sample every Nth row/column of the depth map.",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=20000.0,
        help="Ignore depth values above this limit (mm).",
    )
    parser.add_argument(
        "--camera-name",
        default="CAM_FRONT",
        help="Camera key passed to prepare_mytest_robot_bev.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --dst-root and --robot-bev-out before writing.",
    )
    parser.add_argument(
        "--skip-robot-bev",
        action="store_true",
        help="Only write the mytest source directory, do not run the RobotBEV converter.",
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


def unproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    stride: int,
    max_depth_mm: float,
) -> np.ndarray:
    if stride <= 0:
        raise ValueError("--depth-stride must be positive")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    height, width = depth.shape
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    z_cam = depth[vv, uu].astype(np.float32)
    valid = z_cam > 0
    if max_depth_mm is not None and max_depth_mm > 0:
        valid &= z_cam <= np.float32(max_depth_mm)
    if not np.any(valid):
        raise ValueError("Depth map has no valid points after filtering.")
    x_cam = (uu.astype(np.float32) - np.float32(cx)) * z_cam / np.float32(fx)
    y_cam = (vv.astype(np.float32) - np.float32(cy)) * z_cam / np.float32(fy)
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)[valid]
    return points_cam @ CAMERA_TO_EGO.T


def main() -> int:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    robot_bev_out = args.robot_bev_out.expanduser().resolve()

    for name in ("Depth", "Left", "in.txt"):
        if not (src_root / name).exists():
            raise FileNotFoundError(f"Missing {src_root / name}")

    if args.overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    rgb_dir = dst_root / "rgb"
    pcl_dir = dst_root / "pclCam"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    pcl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_root / "in.txt", dst_root / "in.txt")

    intrinsic = load_intrinsic(src_root / "in.txt")
    depth_files = sorted((src_root / "Depth").glob("*.png"))
    converted = 0
    total_points = 0
    skipped = []
    for depth_path in depth_files:
        stem = depth_path.stem
        match = re.match(r"(\d+)", stem)
        if match is None:
            skipped.append(depth_path.name)
            continue
        frame_id = match.group(1)
        left_path = src_root / "Left" / f"{stem}.jpg"
        if not left_path.exists():
            skipped.append(depth_path.name)
            continue

        rgb_path = rgb_dir / f"{frame_id}.png"
        with Image.open(left_path) as source:
            source.convert("RGB").save(rgb_path)

        depth = np.asarray(Image.open(depth_path))
        if depth.ndim != 2:
            skipped.append(depth_path.name)
            continue
        points = unproject_depth(
            depth, intrinsic, args.depth_stride, args.max_depth_mm
        )
        pcl_path = pcl_dir / (
            f"pointcloud_depth_{frame_id}_LOS_{frame_id}_0.txt"
        )
        np.savetxt(pcl_path, points, fmt="%.3f")
        converted += 1
        total_points += int(points.shape[0])

    if converted == 0:
        raise RuntimeError(f"No frames converted from {src_root}")

    print(
        f"Prepared {converted} frames in {dst_root} "
        f"({total_points} total points, "
        f"avg {total_points / converted:.0f}/frame)"
    )
    if skipped:
        print(f"WARNING: skipped {len(skipped)} frames: {skipped[:10]}")

    if args.skip_robot_bev:
        print("Skipped RobotBEV conversion; source is ready.")
        return 0

    script = Path(__file__).resolve().parent / "prepare_mytest_robot_bev.py"
    cmd = [
        sys.executable,
        str(script),
        "--src-root",
        str(dst_root),
        "--out-root",
        str(robot_bev_out),
        "--dataset-id",
        args.dataset_id,
        "--scene-id",
        args.scene_id,
        "--max-sweeps",
        str(args.max_sweeps),
        "--camera-name",
        args.camera_name,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
