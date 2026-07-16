import pickle

import numpy as np

from mmdet3d.datasets.pipelines.loading import LoadRobotBEVSegmentation
from mmdet3d.datasets.robot_bev_dataset import RobotBEVDataset
from tools.data_converter.robot_bev_converter import convert_split


MAP_CLASSES = ("floor", "carpet", "obstacle", "wall", "furniture", "other")


def _converted_root(canonical_root):
    convert_split(canonical_root, "train", max_sweeps=5)
    return canonical_root


def test_robot_bev_dataset_resolves_root_relative_paths(canonical_root):
    root = _converted_root(canonical_root)
    dataset = RobotBEVDataset(
        ann_file=str(root / "bevfusion_infos_train.pkl"),
        dataset_root=str(root),
        pipeline=[],
        object_classes=[],
        map_classes=MAP_CLASSES,
        modality={"use_camera": True, "use_lidar": True, "use_radar": False},
        test_mode=True,
    )

    data = dataset.get_data_info(1)

    assert data["lidar_path"] == str(root / "scene_a/points/000001.bin")
    assert data["image_paths"] == [str(root / "scene_a/images/000001.png")]
    assert data["bev_mask_path"] == str(root / "scene_a/bev_masks/000001.npy")
    assert data["bev_observed_mask_path"] == str(
        root / "scene_a/bev_observed_masks/000001.npy"
    )
    assert data["class_validity"].tolist() == [1, 1, 1, 1, 1, 1]
    assert data["sweeps"][0]["data_path"] == str(root / "scene_a/points/000000.bin")
    assert data["ann_info"]["gt_labels_3d"].shape == (0,)


def test_load_robot_bev_segmentation_combines_supervision_masks(tmp_path):
    labels = np.zeros((6, 150, 150), dtype=np.uint8)
    labels[0, 0, 0] = 1
    labels[2, 3, 4] = 1
    observed = np.zeros((150, 150), dtype=np.uint8)
    observed[0, 0] = 1
    observed[3, 4] = 1
    regional = np.ones((6, 150, 150), dtype=np.uint8)
    regional[2, 3, 4] = 0
    label_path = tmp_path / "labels.npy"
    observed_path = tmp_path / "observed.npy"
    regional_path = tmp_path / "regional.npy"
    np.save(label_path, labels)
    np.save(observed_path, observed)
    np.save(regional_path, regional)

    result = LoadRobotBEVSegmentation(MAP_CLASSES)(
        {
            "bev_mask_path": str(label_path),
            "bev_observed_mask_path": str(observed_path),
            "bev_supervision_mask_path": str(regional_path),
            "class_validity": np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        }
    )

    assert result["gt_masks_bev"].shape == (6, 150, 150)
    assert result["gt_supervision_mask_bev"].shape == (6, 150, 150)
    assert result["gt_supervision_mask_bev"][0, 0, 0] == 1
    assert result["gt_supervision_mask_bev"][1, 0, 0] == 0
    assert result["gt_supervision_mask_bev"][2, 3, 4] == 0


def test_robot_bev_dataset_metadata_matches_current_infos(canonical_root):
    root = _converted_root(canonical_root)
    with (root / "bevfusion_infos_train.pkl").open("rb") as handle:
        payload = pickle.load(handle)

    dataset = RobotBEVDataset(
        ann_file=str(root / "bevfusion_infos_train.pkl"),
        dataset_root=str(root),
        pipeline=[],
        object_classes=[],
        map_classes=payload["metadata"]["map_classes"],
        test_mode=True,
    )

    assert dataset.version == "robot-bev-v3"
    assert tuple(dataset.map_classes) == MAP_CLASSES
    assert len(dataset) == 2
