import argparse
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare single-camera depth point clouds for BEVFusion inference."
    )
    parser.add_argument(
        "--src-root",
        default="data/mytest/data",
        help="Input folder that contains rgb/, pclCam/ and in.txt.",
    )
    parser.add_argument(
        "--out-root",
        default="data/mytest/processed",
        help="Output folder for converted images, point bins and info pkl.",
    )
    parser.add_argument(
        "--camera2lidar",
        default=None,
        help=(
            "Path to a 4x4 or 3x4 camera-to-LiDAR/vehicle matrix. "
            "Column-vector convention: p_lidar = R @ p_camera + t."
        ),
    )
    parser.add_argument(
        "--point-scale",
        type=float,
        default=0.001,
        help="Scale applied to txt point coordinates.",
    )
    parser.add_argument(
        "--points-coord",
        choices=("camera", "ego"),
        default="camera",
        help=(
            "Coordinate frame of pclCam txt points. Use 'camera' to transform "
            "points by --camera2lidar; use 'ego' when points are already in "
            "the LiDAR/ego frame."
        ),
    )
    parser.add_argument(
        "--info-name",
        default="mytest_infos_val.pkl",
        help="Output info pkl name under out-root.",
    )
    parser.add_argument(
        "--camera-name",
        default="CAM_FRONT",
        help="Camera key written into the info file.",
    )
    return parser.parse_args()


def load_intrinsic(path):
    text = Path(path).read_text()
    values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]
    if len(values) < 4:
        raise ValueError(f"Expected fx, fy, cx, cy in {path}, got: {text!r}")
    fx, fy, cx, cy = values[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def load_camera2lidar(path):
    if path is None:
        print("WARNING: --camera2lidar not provided; using identity transform.")
        return np.eye(4, dtype=np.float32)

    mat = np.loadtxt(path, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    raise ValueError(f"{path} must contain a 4x4 or 3x4 matrix, got shape {mat.shape}")


def rotation_matrix_to_quaternion(rot):
    """Return a pyquaternion-compatible [w, x, y, z] quaternion."""
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return (quat / np.linalg.norm(quat)).tolist()


def frame_id_from_rgb(path):
    return int(path.stem)


def frame_id_from_pointcloud(path):
    match = re.search(r"_LOS_(\d+)_", path.name)
    if match is None:
        return None
    return int(match.group(1))


def read_points_txt(path, scale):
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"{path} must contain at least xyz columns")
    return points[:, :3] * scale


def transform_points(points_cam, camera2lidar):
    rot = camera2lidar[:3, :3]
    trans = camera2lidar[:3, 3]
    return points_cam @ rot.T + trans


def make_empty_annos():
    return dict(
        gt_boxes=np.zeros((0, 7), dtype=np.float32),
        gt_names=np.array([], dtype=object),
        gt_velocity=np.zeros((0, 2), dtype=np.float32),
        num_lidar_pts=np.zeros((0,), dtype=np.int64),
        num_radar_pts=np.zeros((0,), dtype=np.int64),
        valid_flag=np.zeros((0,), dtype=bool),
    )


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    image_out = out_root / "images"
    points_out = out_root / "points"
    image_out.mkdir(parents=True, exist_ok=True)
    points_out.mkdir(parents=True, exist_ok=True)

    intrinsic = load_intrinsic(src_root / "in.txt")
    camera2lidar = load_camera2lidar(args.camera2lidar)
    cam_rot = camera2lidar[:3, :3].astype(np.float32)
    cam_trans = camera2lidar[:3, 3].astype(np.float32)
    cam_quat = rotation_matrix_to_quaternion(cam_rot)

    point_files = {}
    for path in (src_root / "pclCam").glob("*.txt"):
        frame_id = frame_id_from_pointcloud(path)
        if frame_id is not None:
            point_files[frame_id] = path

    infos = []
    prev_token = ""
    rgb_files = sorted((src_root / "rgb").glob("*.png"), key=frame_id_from_rgb)
    for rgb_path in rgb_files:
        frame_id = frame_id_from_rgb(rgb_path)
        pcl_path = point_files.get(frame_id)
        if pcl_path is None:
            print(f"WARNING: skip frame {frame_id}, no matching pclCam txt found.")
            continue

        token = f"mytest_{frame_id:06d}"
        dst_img = image_out / f"{frame_id:06d}.png"
        dst_bin = points_out / f"{frame_id:06d}.bin"

        image = Image.open(rgb_path).convert("RGB")
        image.save(dst_img)

        points = read_points_txt(pcl_path, args.point_scale)
        if args.points_coord == "camera":
            points_lidar = transform_points(points, camera2lidar)
        else:
            points_lidar = points
        attrs = np.zeros((points_lidar.shape[0], 2), dtype=np.float32)
        points_5d = np.concatenate([points_lidar.astype(np.float32), attrs], axis=1)
        points_5d.tofile(dst_bin)

        timestamp = int(frame_id)
        cam_info = dict(
            data_path=str(dst_img),
            type=args.camera_name,
            sample_data_token=f"{token}_{args.camera_name}",
            sensor2ego_translation=cam_trans.tolist(),
            sensor2ego_rotation=cam_quat,
            ego2global_translation=[0.0, 0.0, 0.0],
            ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
            timestamp=timestamp,
            cam_intrinsic=intrinsic,
            sensor2lidar_rotation=cam_rot,
            sensor2lidar_translation=cam_trans,
        )
        info = dict(
            lidar_path=str(dst_bin),
            token=token,
            sweeps=[],
            cams={args.camera_name: cam_info},
            radars={},
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
            ego2global_translation=[0.0, 0.0, 0.0],
            ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
            timestamp=timestamp,
            prev_token=prev_token,
            location="mytest",
            **make_empty_annos(),
        )
        infos.append(info)
        prev_token = token

    data = dict(infos=infos, metadata=dict(version="mytest", camera=args.camera_name))
    info_path = out_root / args.info_name
    with info_path.open("wb") as f:
        pickle.dump(data, f)

    latest = out_root / "mytest_infos_test.pkl"
    if latest != info_path:
        shutil.copyfile(info_path, latest)

    print(f"Prepared {len(infos)} frames")
    print(f"Images: {image_out}")
    print(f"Point bins: {points_out}")
    print(f"Info: {info_path}")


if __name__ == "__main__":
    main()

# 选择 1：先不传外参，脚本会用单位矩阵
# python tools/prepare_mytest_data.py \
#   --src-root data/mytest/data \
#   --out-root data/mytest/processed \
#   --point-scale 0.001

# 选择 2：有相机坐标到车体/LiDAR坐标的外参
# python tools/prepare_mytest_data.py \
#   --src-root data/mytest/data \
#   --out-root data/mytest/processed \
#   --camera2lidar data/mytest/camera2lidar.txt \
#   --point-scale 0.001

# Option 3: pclCam points 已经是自车轴线
# Keep --camera2lidar for camera calibration, but do not transform points.
# python tools/prepare_mytest_data.py \
#   --src-root data/mytest/data \
#   --out-root data/mytest/processed \
#   --camera2lidar data/mytest/camera2lidar.txt \
#   --points-coord ego \
#   --point-scale 0.001
