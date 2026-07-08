import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert robot BEV segmentation infos to BEVFusion format."
    )
    parser.add_argument(
        "--src-root",
        default="data/test_dataset/test_dataset/robot_closed_loop_multi",
        help="Robot dataset root containing robot_infos_train.pkl/val.pkl.",
    )
    parser.add_argument(
        "--out-prefix",
        default="robot_seg_infos",
        help="Output pkl prefix written under src-root.",
    )
    parser.add_argument(
        "--camera-name",
        default="CAM_FRONT",
        help="Camera key written into the converted info file.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=5,
        help="Number of previous LiDAR sweeps to attach per frame.",
    )
    return parser.parse_args()


def rotation_matrix_to_quaternion(rot):
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
    quat /= np.linalg.norm(quat)
    return quat.tolist()


def load_infos(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["infos"], data.get("metadata", {})


def normalize_matrix(value):
    mat = np.asarray(value, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got {mat.shape}")
    return mat


def normalize_intrinsic(value):
    mat = np.asarray(value, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got {mat.shape}")
    return mat


def resolve_dataset_path(src_root, raw_path):
    raw = Path(str(raw_path).replace("\\", "/"))
    if raw.is_absolute():
        return raw.as_posix()

    parts = raw.parts
    root_parts = src_root.parts
    if len(parts) >= 2 and parts[:2] == ("test_dataset", "robot_closed_loop_multi"):
        return (src_root / Path(*parts[2:])).as_posix()
    if len(parts) >= 1 and parts[0] in root_parts:
        return raw.as_posix()

    return (src_root / raw).as_posix()


def scene_name(info):
    return Path(str(info["lidar_path"]).replace("\\", "/")).parts[-3]


def frame_id(info):
    return int(Path(info["lidar_path"]).stem)


def make_sweeps(info, scene_infos, src_root, max_sweeps):
    cur_frame = frame_id(info)
    cur_lidar2base = normalize_matrix(info["lidar2base"])
    cur_base2map = normalize_matrix(info["T_map_base"])
    cur_lidar2map = cur_base2map @ cur_lidar2base
    cur_map2lidar = np.linalg.inv(cur_lidar2map)

    sweeps = []
    for hist_info in reversed(scene_infos):
        if frame_id(hist_info) >= cur_frame:
            continue
        hist_lidar2base = normalize_matrix(hist_info["lidar2base"])
        hist_base2map = normalize_matrix(hist_info["T_map_base"])
        hist_lidar2map = hist_base2map @ hist_lidar2base
        hist_lidar2cur_lidar = cur_map2lidar @ hist_lidar2map
        sweeps.append(
            dict(
                data_path=resolve_dataset_path(src_root, hist_info["lidar_path"]),
                timestamp=hist_info["timestamp"],
                sensor2lidar_rotation=hist_lidar2cur_lidar[:3, :3].astype(np.float32),
                sensor2lidar_translation=hist_lidar2cur_lidar[:3, 3].astype(
                    np.float32
                ),
            )
        )
        if len(sweeps) >= max_sweeps:
            break
    return sweeps


def convert_info(info, all_by_scene, src_root, camera_name, max_sweeps):
    cam2base = normalize_matrix(info["camera2base"])
    lidar2base = normalize_matrix(info["lidar2base"])
    base2map = normalize_matrix(info["T_map_base"])
    cam2lidar = np.linalg.inv(lidar2base) @ cam2base
    scene = scene_name(info)
    token = f"{scene}_{frame_id(info):06d}"
    prev_token = ""
    if info.get("prev_token"):
        prev_token = f"{scene}_{int(info['prev_token']):06d}"

    cam_info = dict(
        data_path=resolve_dataset_path(src_root, info["image_path"]),
        type=camera_name,
        sample_data_token=f"{token}_{camera_name}",
        sensor2ego_translation=cam2base[:3, 3].tolist(),
        sensor2ego_rotation=rotation_matrix_to_quaternion(cam2base[:3, :3]),
        ego2global_translation=base2map[:3, 3].tolist(),
        ego2global_rotation=rotation_matrix_to_quaternion(base2map[:3, :3]),
        timestamp=info["timestamp"],
        cam_intrinsic=normalize_intrinsic(info["cam_intrinsic"]),
        sensor2lidar_rotation=cam2lidar[:3, :3].astype(np.float32),
        sensor2lidar_translation=cam2lidar[:3, 3].astype(np.float32),
    )

    converted = dict(
        lidar_path=resolve_dataset_path(src_root, info["lidar_path"]),
        token=token,
        sweeps=make_sweeps(info, all_by_scene[scene], src_root, max_sweeps),
        cams={camera_name: cam_info},
        radars={},
        lidar2ego_translation=lidar2base[:3, 3].tolist(),
        lidar2ego_rotation=rotation_matrix_to_quaternion(lidar2base[:3, :3]),
        ego2global_translation=base2map[:3, 3].tolist(),
        ego2global_rotation=rotation_matrix_to_quaternion(base2map[:3, :3]),
        timestamp=info["timestamp"],
        prev_token=prev_token,
        location="robot",
        bev_mask_path=resolve_dataset_path(src_root, info["bev_mask_path"]),
        gt_boxes=np.asarray(info["gt_boxes"], dtype=np.float32),
        gt_names=np.asarray(info["gt_names"], dtype=object),
        gt_velocity=np.asarray(info["gt_velocity"], dtype=np.float32),
        num_lidar_pts=np.asarray(info["num_lidar_pts"], dtype=np.int64),
        num_radar_pts=np.asarray(info["num_radar_pts"], dtype=np.int64),
        valid_flag=np.asarray(info["valid_flag"], dtype=bool),
    )
    return converted


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    train_infos, train_meta = load_infos(src_root / "robot_infos_train.pkl")
    val_infos, val_meta = load_infos(src_root / "robot_infos_val.pkl")

    all_by_scene = {}
    for info in train_infos + val_infos:
        all_by_scene.setdefault(scene_name(info), []).append(info)
    for infos in all_by_scene.values():
        infos.sort(key=frame_id)

    metadata = dict(train_meta)
    metadata.update(val_meta)
    metadata.update(
        version="robot-seg",
        camera=args.camera_name,
        source="robot_closed_loop_multi",
    )

    for split, infos in [("train", train_infos), ("val", val_infos)]:
        converted = [
            convert_info(info, all_by_scene, src_root, args.camera_name, args.max_sweeps)
            for info in infos
        ]
        out_path = src_root / f"{args.out_prefix}_{split}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(dict(infos=converted, metadata=metadata), f)
        print(f"Saved {len(converted)} {split} infos to {out_path}")


if __name__ == "__main__":
    main()






# --src-root 是机器人虚拟数据集的根目录，也就是里面直接包含这些文件/目录的那一层
# python selffile\convert_robot_seg_data.py --src-root data\test_dataset\test_dataset\robot_closed_loop_multi