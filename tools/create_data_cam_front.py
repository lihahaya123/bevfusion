import argparse
from os import path as osp

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from data_converter.create_gt_database import create_groundtruth_database
from data_converter.nuscenes_converter import get_available_scenes, obtain_sensor2top
from mmdet3d.datasets import NuScenesDataset


def create_cam_front_nuscenes_infos(
    root_path,
    info_prefix,
    version="v1.0-trainval",
    max_sweeps=10,
):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError(f"unknown nuScenes version: {version}")

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    )
    val_scenes = set(
        available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes
    )

    test = "test" in version
    train_infos, val_infos = _fill_cam_front_trainval_infos(
        nusc, train_scenes, val_scenes, test=test, max_sweeps=max_sweeps
    )

    metadata = dict(version=version, camera="CAM_FRONT")
    if test:
        data = dict(infos=train_infos, metadata=metadata)
        info_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        mmcv.dump(data, info_path)
        print(f"saved {len(train_infos)} test samples to {info_path}")
    else:
        train_data = dict(infos=train_infos, metadata=metadata)
        train_info_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        mmcv.dump(train_data, train_info_path)

        val_data = dict(infos=val_infos, metadata=metadata)
        val_info_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        mmcv.dump(val_data, val_info_path)
        print(f"saved {len(train_infos)} train samples to {train_info_path}")
        print(f"saved {len(val_infos)} val samples to {val_info_path}")


def _fill_cam_front_trainval_infos(
    nusc,
    train_scenes,
    val_scenes,
    test=False,
    max_sweeps=10,
):
    train_infos = []
    val_infos = []
    token2idx = {}

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "radars": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            "prev_token": sample["prev"],
        }

        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        info["location"] = log["location"]

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        cam = "CAM_FRONT"
        cam_token = sample["data"][cam]
        _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(
            nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
        )
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info["cams"].update({cam: cam_info})

        sd_rec = nusc.get("sample_data", lidar_token)
        sweeps = []
        while len(sweeps) < max_sweeps:
            if sd_rec["prev"] != "":
                sweep = obtain_sensor2top(
                    nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps

        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations],
                dtype=bool,
            ).reshape(-1)

            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = (
                    velo
                    @ np.linalg.inv(e2g_r_mat).T
                    @ np.linalg.inv(l2e_r_mat).T
                )
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i, name in enumerate(names):
                if name in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[name]
            names = np.array(names)
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations]
            )
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations]
            )
            info["valid_flag"] = valid_flag

        if sample["scene_token"] in train_scenes:
            train_infos.append(info)
            token2idx[info["token"]] = ("train", len(train_infos) - 1)
        else:
            val_infos.append(info)
            token2idx[info["token"]] = ("val", len(val_infos) - 1)

    for infos, split in [(train_infos, "train"), (val_infos, "val")]:
        for info in infos:
            prev_token = info["prev_token"]
            if prev_token == "":
                info["prev"] = -1
            else:
                prev_split, prev_idx = token2idx[prev_token]
                assert prev_split == split
                info["prev"] = prev_idx

    return train_infos, val_infos


def nuscenes_cam_front_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    create_gt_database=False,
):
    create_cam_front_nuscenes_infos(
        root_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps,
    )

    if create_gt_database:
        create_groundtruth_database(
            dataset_name,
            root_path,
            info_prefix,
            f"{out_dir}/{info_prefix}_infos_train.pkl",
        )


parser = argparse.ArgumentParser(description="CAM_FRONT nuScenes data converter")
parser.add_argument("dataset", metavar="nuscenes", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/nuscenes",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/nuscenes",
    required=False,
    help="output directory",
)
parser.add_argument("--extra-tag", type=str, default="nuscenes_cam_front")
parser.add_argument(
    "--create-gt-database",
    default=False,
    action="store_true",
    help="also create GT database for training",
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.dataset != "nuscenes":
        raise ValueError("Only nuScenes is supported.")

    if args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_cam_front_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            create_gt_database=args.create_gt_database,
        )

        test_version = f"{args.version}-test"
        nuscenes_cam_front_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            create_gt_database=False,
        )
    else:
        nuscenes_cam_front_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            create_gt_database=args.create_gt_database,
        )


# Example:
# python tools/create_data_cam_front.py nuscenes --root-path ./data/nuscenes_1 --out-dir ./data/nuscenes_1 --extra-tag nuscenes_cam_front --version v1.0-mini --max-sweeps 10
