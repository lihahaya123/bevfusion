import copy
import os
from os import path as osp
from typing import Any, Dict, Mapping, Optional, Sequence

import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class RobotBEVDataset(Custom3DDataset):
    """Dataset for canonical Robot BEV v3 converted BEVFusion infos."""

    MAP_CLASSES = (
        "floor",
        "carpet",
        "obstacle",
        "wall",
        "furniture",
        "other",
    )
    CLASSES = ()

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=False,
        test_mode=False,
        use_valid_flag=False,
        with_velocity=False,
    ) -> None:
        if dataset_root is None:
            dataset_root = osp.dirname(osp.abspath(ann_file))
        self.load_interval = load_interval
        self.map_classes = tuple(map_classes or self.MAP_CLASSES)
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes or [],
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def load_annotations(self, ann_file):
        payload = mmcv.load(ann_file)
        self.metadata = payload["metadata"]
        self.version = self.metadata.get("version", "robot-bev-v3")
        source_classes = tuple(self.metadata.get("map_classes", ()))
        if source_classes and source_classes != self.map_classes:
            raise ValueError(
                f"map_classes mismatch: ann_file has {source_classes}, "
                f"dataset config has {self.map_classes}"
            )
        infos = list(payload["infos"])
        return infos[:: self.load_interval]

    def get_cat_ids(self, idx):
        return []

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info["token"],
            lidar_path=self._resolve_path(info["lidar_path"]),
            sweeps=self._resolve_sweeps(info.get("sweeps", [])),
            timestamp=int(info["timestamp"]),
            bev_mask_path=self._resolve_path(info["bev_mask_path"]),
            bev_observed_mask_path=self._resolve_path(
                info["bev_observed_mask_path"]
            ),
            bev_supervision_mask_path=self._resolve_optional_path(
                info.get("bev_supervision_mask_path")
            ),
            class_validity=np.asarray(info["class_validity"], dtype=np.uint8).copy(),
        )
        for key in ("depth_path", "semantic_path"):
            value = self._resolve_optional_path(info.get(key))
            if value is not None:
                data[key] = value

        data["ego2global"] = self._quat_pose(
            info["ego2global_rotation"], info["ego2global_translation"]
        )
        data["lidar2ego"] = self._quat_pose(
            info["lidar2ego_rotation"], info["lidar2ego_translation"]
        )

        if self.modality is not None and self.modality.get("use_camera", False):
            self._fill_camera_fields(data, info["cams"])

        data["ann_info"] = self.get_ann_info(index)
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag and "valid_flag" in info:
            mask = np.asarray(info["valid_flag"], dtype=bool)
        else:
            mask = np.asarray(info.get("num_lidar_pts", [])) > 0

        gt_bboxes_3d = np.asarray(
            info.get("gt_boxes", np.zeros((0, 7))),
            dtype=np.float32,
        )
        gt_names_3d = np.asarray(info.get("gt_names", []))
        if mask.size:
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names_3d = gt_names_3d[mask]
        gt_labels_3d = np.array(
            [
                self.CLASSES.index(name) if name in self.CLASSES else -1
                for name in gt_names_3d
            ],
            dtype=np.int64,
        )

        if self.with_velocity:
            gt_velocity = np.asarray(
                info.get("gt_velocity", np.zeros((len(gt_bboxes_3d), 2))),
                dtype=np.float32,
            )
            if mask.size:
                gt_velocity = gt_velocity[mask]
            nan_mask = np.isnan(gt_velocity[:, 0]) if len(gt_velocity) else []
            if len(gt_velocity):
                gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1] if gt_bboxes_3d.ndim == 2 else 7,
            origin=(0.5, 0.5, 0),
        ).convert_to(self.box_mode_3d)
        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)
        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)
        valid_pixels = torch.zeros(num_classes)

        for result in results:
            pred = result["masks_bev"].detach().reshape(num_classes, -1)
            label = result["gt_masks_bev"].detach().bool().reshape(num_classes, -1)
            mask = result.get("gt_supervision_mask_bev")
            if mask is None:
                mask = torch.ones_like(label, dtype=torch.bool)
            else:
                mask = mask.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds.to(pred.device)
            label = label[:, :, None]
            mask_t = mask[:, :, None]
            valid_pixels += mask.sum(dim=1).cpu()
            tp += ((pred & label) & mask_t).sum(dim=1).cpu()
            fp += ((pred & ~label) & mask_t).sum(dim=1).cpu()
            fn += ((~pred & label) & mask_t).sum(dim=1).cpu()

        ious = tp / (tp + fp + fn + 1e-7)
        metrics = {}
        valid_classes = valid_pixels > 0
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/valid_pixels"] = valid_pixels[index].item()
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        if valid_classes.any():
            metrics["map/mean/iou@0.50"] = ious[valid_classes, 3].mean().item()
            metrics["map/mean/iou@max"] = (
                ious[valid_classes].max(dim=1).values.mean().item()
            )
        else:
            metrics["map/mean/iou@0.50"] = 0.0
            metrics["map/mean/iou@max"] = 0.0
        return metrics

    def evaluate(self, results, **kwargs):
        if not results:
            return {}
        metrics = {}
        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))
        return metrics

    def _resolve_path(self, value: str) -> str:
        path = os.fspath(value)
        if osp.isabs(path):
            return path
        return osp.join(self.dataset_root, path)

    def _resolve_optional_path(self, value: Optional[str]) -> Optional[str]:
        return None if value is None else self._resolve_path(value)

    def _resolve_sweeps(self, sweeps: Sequence[Mapping[str, Any]]):
        resolved = []
        for sweep in sweeps:
            item = copy.deepcopy(dict(sweep))
            item["data_path"] = self._resolve_path(item["data_path"])
            resolved.append(item)
        return resolved

    @staticmethod
    def _quat_pose(rotation, translation) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = Quaternion(rotation).rotation_matrix
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
        return matrix

    def _fill_camera_fields(
        self,
        data: Dict[str, Any],
        cams: Mapping[str, Any],
    ) -> None:
        data["image_paths"] = []
        data["lidar2camera"] = []
        data["lidar2image"] = []
        data["camera2ego"] = []
        data["camera_intrinsics"] = []
        data["camera2lidar"] = []

        for _, camera_info in cams.items():
            data["image_paths"].append(self._resolve_path(camera_info["data_path"]))
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4, dtype=np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            data["lidar2camera"].append(lidar2camera_rt.T)

            camera_intrinsics = np.eye(4, dtype=np.float32)
            camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
            data["camera_intrinsics"].append(camera_intrinsics)
            data["lidar2image"].append(camera_intrinsics @ lidar2camera_rt.T)

            camera2ego = self._quat_pose(
                camera_info["sensor2ego_rotation"],
                camera_info["sensor2ego_translation"],
            )
            data["camera2ego"].append(camera2ego)

            camera2lidar = np.eye(4, dtype=np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            data["camera2lidar"].append(camera2lidar)
