import copy
import os
from os import path as osp
from typing import Any, Dict, Mapping, Optional, Sequence

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class RobotBEVDataset(Custom3DDataset):
    """Dataset for canonical Robot BEV converted BEVFusion infos."""

    MAP_CLASSES = (
        "floor",
        "carpet",
        "wall",
        "furniture",
        "door",
        "clutter",
    )
    CLASSES = ()

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        map_classes=None,
        load_interval=1,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=False,
        test_mode=False,
    ) -> None:
        if dataset_root is None:
            dataset_root = osp.dirname(osp.abspath(ann_file))
        self.load_interval = load_interval
        self.map_classes = tuple(map_classes or self.MAP_CLASSES)
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=None,
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
        self.version = self.metadata.get("version", "robot-bev-v4")
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

        return data

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)
        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)
        valid_pixels = torch.zeros(num_classes)
        boundary_pred = torch.zeros(num_classes)
        boundary_gt = torch.zeros(num_classes)
        boundary_pred_matched = torch.zeros(num_classes)
        boundary_gt_matched = torch.zeros(num_classes)

        for result in results:
            pred_map = result["masks_bev"].detach()
            label_map = result["gt_masks_bev"].detach().bool()
            mask = result.get("gt_supervision_mask_bev")
            if mask is None:
                mask_map = torch.ones_like(label_map, dtype=torch.bool)
            else:
                mask_map = mask.detach().bool()

            pred = pred_map.reshape(num_classes, -1)
            label = label_map.reshape(num_classes, -1)
            mask = mask_map.reshape(num_classes, -1)
            pred = pred[:, :, None] >= thresholds.to(pred.device)
            label = label[:, :, None]
            mask_t = mask[:, :, None]
            valid_pixels += mask.sum(dim=1).cpu()
            tp += ((pred & label) & mask_t).sum(dim=1).cpu()
            fp += ((pred & ~label) & mask_t).sum(dim=1).cpu()
            fn += ((~pred & label) & mask_t).sum(dim=1).cpu()

            boundary_counts = self._boundary_counts(
                pred_map >= 0.5,
                label_map,
                mask_map,
                tolerance=1,
            )
            boundary_pred += boundary_counts["pred"]
            boundary_gt += boundary_counts["gt"]
            boundary_pred_matched += boundary_counts["pred_matched"]
            boundary_gt_matched += boundary_counts["gt_matched"]

        ious = tp / (tp + fp + fn + 1e-7)
        precision_50 = tp[:, 3] / (tp[:, 3] + fp[:, 3] + 1e-7)
        recall_50 = tp[:, 3] / (tp[:, 3] + fn[:, 3] + 1e-7)
        f1_50 = (
            2
            * precision_50
            * recall_50
            / (precision_50 + recall_50 + 1e-7)
        )
        boundary_precision = boundary_pred_matched / (boundary_pred + 1e-7)
        boundary_recall = boundary_gt_matched / (boundary_gt + 1e-7)
        boundary_f1 = (
            2
            * boundary_precision
            * boundary_recall
            / (boundary_precision + boundary_recall + 1e-7)
        )
        metrics = {}
        valid_classes = valid_pixels > 0
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/valid_pixels"] = valid_pixels[index].item()
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            metrics[f"map/{name}/precision@0.50"] = precision_50[index].item()
            metrics[f"map/{name}/recall@0.50"] = recall_50[index].item()
            metrics[f"map/{name}/f1@0.50"] = f1_50[index].item()
            metrics[f"map/{name}/boundary_precision@0.50"] = (
                boundary_precision[index].item()
            )
            metrics[f"map/{name}/boundary_recall@0.50"] = (
                boundary_recall[index].item()
            )
            metrics[f"map/{name}/boundary_f1@0.50"] = boundary_f1[index].item()
            metrics[f"map/{name}/boundary_gt_pixels"] = boundary_gt[index].item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        if valid_classes.any():
            mean_iou_50 = ious[valid_classes, 3].mean().item()
            mean_iou_max = ious[valid_classes].max(dim=1).values.mean().item()
            mean_precision_50 = precision_50[valid_classes].mean().item()
            mean_recall_50 = recall_50[valid_classes].mean().item()
            mean_f1_50 = f1_50[valid_classes].mean().item()
        else:
            mean_iou_50 = 0.0
            mean_iou_max = 0.0
            mean_precision_50 = 0.0
            mean_recall_50 = 0.0
            mean_f1_50 = 0.0
        boundary_valid_classes = boundary_gt > 0
        if boundary_valid_classes.any():
            mean_boundary_f1 = boundary_f1[boundary_valid_classes].mean().item()
        else:
            mean_boundary_f1 = 0.0
        metrics["map/mean/iou@0.50"] = mean_iou_50
        metrics["map/mean/iou@max"] = mean_iou_max
        metrics["map/mean/precision@0.50"] = mean_precision_50
        metrics["map/mean/recall@0.50"] = mean_recall_50
        metrics["map/mean/f1@0.50"] = mean_f1_50
        metrics["map/mean/boundary_f1@0.50"] = mean_boundary_f1
        # MMCV EvalHook uses the save_best metric name in checkpoint filenames.
        # Keep slash-free aliases so best checkpoint paths are safe on disk.
        metrics["robotbev_map_iou_50"] = mean_iou_50
        metrics["robotbev_map_iou_max"] = mean_iou_max
        metrics["robotbev_map_f1_50"] = mean_f1_50
        metrics["robotbev_boundary_f1_50"] = mean_boundary_f1
        return metrics

    @staticmethod
    def _boundary_counts(prediction, target, supervision, tolerance=1):
        prediction = prediction.bool()
        target = target.bool()
        supervision = supervision.bool()

        def erode(mask, radius):
            kernel = radius * 2 + 1
            inverted = (~mask).float().unsqueeze(0)
            dilated_inverted = F.max_pool2d(
                inverted, kernel_size=kernel, stride=1, padding=radius
            )
            return ~(dilated_inverted.squeeze(0).bool())

        def boundary(mask):
            return mask & ~erode(mask, 1)

        def dilate(mask, radius):
            kernel = radius * 2 + 1
            return F.max_pool2d(
                mask.float().unsqueeze(0),
                kernel_size=kernel,
                stride=1,
                padding=radius,
            ).squeeze(0).bool()

        valid_core = erode(supervision, tolerance)
        pred_boundary = boundary(prediction) & valid_core
        gt_boundary = boundary(target) & valid_core
        pred_matched = pred_boundary & dilate(gt_boundary, tolerance)
        gt_matched = gt_boundary & dilate(pred_boundary, tolerance)
        return {
            "pred": pred_boundary.reshape(prediction.shape[0], -1).sum(1).cpu(),
            "gt": gt_boundary.reshape(prediction.shape[0], -1).sum(1).cpu(),
            "pred_matched": pred_matched.reshape(prediction.shape[0], -1)
            .sum(1)
            .cpu(),
            "gt_matched": gt_matched.reshape(prediction.shape[0], -1).sum(1).cpu(),
        }

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
