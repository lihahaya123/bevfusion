"""测试局部点云稀疏时 BEVFusion 对退化区域的 BEV 语义补充能力。

点云退化方式
============
1. ``region_mask``：在指定 LiDAR 坐标范围内随机稀疏采样，默认保留 10%
   的点，可用 ``region_keep_ratio`` 调整。区域边界包含在内，x/y 单位为米，
   z 不受限制；数据管线加载的所有 sweep 都一起处理。
2. ``full_lidar_remove``：删除整帧全部点云。
3. ``density_reduce``：随机保留 ``density_ratio`` 比例的点。
4. ``random_drop``：随机删除 ``drop_ratio`` 比例的点。

``region_mask`` 示例区域 ``x=[1,2], y=[-0.5,0.5]`` 是 1 m x 1 m 区域。
可视化中的红框和 ``LiDAR sparse`` 只标识稀疏区域，不参与模型推理。

每个样本的四组对照
==================
``original``
    完整点云 + 相机，是正常融合性能基准；红框内的点没有删除。
``degraded``
    退化点云 + 相机，是需要评估的实际结果。
``camera_only``
    空点云 + 相机，用于观察当前融合模型在没有 LiDAR 信息时的表现。
``camera_ablated_degraded``
    退化点云 + 相机 BEV 特征置零，用于估计残余 LiDAR 自身的贡献。

指标及区域
==========
预测概率统一使用 ``map_threshold`` 二值化。对每个类别统计：

* IoU = TP / (TP + FP + FN)：主要指标，同时惩罚漏检和误检。
* Precision = TP / (TP + FP)：预测出的区域有多少正确；低值可能表示乱补。
* Recall = TP / (TP + FN)：GT 区域有多少被补出；低值表示补充不足。
* F1 = 2 * Precision * Recall / (Precision + Recall)：精确率与召回率的综合值。
* ``mean_*``：有效类别上的平均值。

``results.json`` 同时报告三个评价区域：

* ``full``：完整 BEV。
* ``missing``：为兼容原结果保留的字段名，实际表示局部稀疏区域；判断补充
  能力时应优先查看。
* ``retained``：局部稀疏区域以外的区域。

全数据集 ``summary`` 先累加所有样本的 TP/FP/FN，再计算汇总指标，并非简单
平均每帧 mIoU。关键比较量（O=original，D=degraded，C=camera_only，
A=camera_ablated_degraded，均指 missing 区域 mIoU）为：

* ``degraded_minus_original = D - O``：点云缺失造成的性能变化，越接近 0
  表示越鲁棒，明显小于 0 表示退化严重。
* ``degraded_minus_camera_only = D - C``：残余点云相对纯相机的贡献。
* ``camera_supplement_gain = D - A``：相机在点云残缺条件下带来的增益；
  稳定大于 0 才能作为相机具有补充作用的证据。
* ``recovery_ratio = (D - A) / (O - A)``：相机恢复正常融合能力的比例；
  0 表示没有恢复，1 表示恢复到完整点云基准附近，负值表示产生负作用。

判断建议
========
先确认 O 本身有效且 D 相比 O 确实下降，再重点比较 D 与 A。只有当多帧汇总的
``camera_supplement_gain`` 稳定为正、``recovery_ratio`` 较高，同时红框内 D
比 A 更接近 GT，才能认为存在可靠补充能力。单帧结果只适合排查和展示。

输出
====
``output_dir/results.json`` 保存逐帧与汇总指标；设置 ``show_dir`` 和
``viz_samples`` 后，前 N 帧只保存 GT、完整点云预测、局部稀疏点云预测、
局部稀疏点云且相机 BEV 特征置零预测的 PNG，不保存 NPY。PNG 使用与
``map_threshold`` 相同的显示阈值。

示例（``num_samples=0`` 表示测试全部数据，只保存前 20 帧可视化）：

.. code-block:: bash

python tools/test_lidar_degradation.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  data/replica_base_train/best_robotbev_map_iou_max_epoch_2.pth \
  --degradation region_mask \
  --region '{"x_min":1,"x_max":2,"y_min":-0.5,"y_max":0.5}' \
  --region-keep-ratio 0.1 \
  --device cuda:0 \
  --num-samples 200 \
  --output-dir ./lidar_degradation_results_sparse10 \
  --data-root data/replica_robot_bev_v4 \
  --viz-samples 20 \
  --show-dir ./lidar_degradation_results_sparse10/visualizations

"""

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

# -------------------------------------------------------------------------
# 与原始 tools/test.py 保持一致：项目根目录
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import cv2

from torchpack.utils.config import configs

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.core.utils import visualize_map, visualize_map_scores
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet.apis import set_random_seed


DEFAULT_REGION = {
    "x_min": 1.0,
    "x_max": 2.0,
    "y_min": -0.5,
    "y_max": 0.5,
}


# =========================================================================
# 参数
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test BEVFusion robustness to LiDAR degradation"
    )

    # 同时兼容：
    #   python xxx.py config.py checkpoint.pth
    # 和代码二原来的：
    #   python xxx.py --config config.py --checkpoint checkpoint.pth
    parser.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        help="config file path (positional, optional)",
    )
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="checkpoint file (positional, optional)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config file path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint file",
    )

    parser.add_argument(
        "--degradation",
        type=str,
        choices=[
            "full_lidar_remove",
            "region_mask",
            "density_reduce",
            "random_drop",
        ],
        default="region_mask",
        help="LiDAR degradation type",
    )

    # 同时兼容 argparse 常见的连字符和代码二原来的下划线写法
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="./lidar_degradation_results",
        help="output directory",
    )

    parser.add_argument(
        "--num-samples",
        "--num_samples",
        dest="num_samples",
        type=int,
        default=0,
        help="number of test samples; <= 0 means the complete test set",
    )

    # 为兼容代码二保留；当前新版主要保存 numpy，因此这个参数只控制
    # 后续可视化/保存数量，不影响核心 forward。
    parser.add_argument(
        "--viz-samples",
        "--viz_samples",
        dest="viz_samples",
        type=int,
        default=0,
        help="number of samples to save visualization/numpy results for",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help=(
            "region JSON; region_mask defaults to the central 1 m x 1 m area: "
            '{"x_min":1,"x_max":2,"y_min":-0.5,"y_max":0.5}'
        ),
    )

    parser.add_argument(
        "--region-keep-ratio",
        "--region_keep_ratio",
        dest="region_keep_ratio",
        type=float,
        default=0.1,
        help=(
            "fraction of points randomly retained inside --region for "
            "region_mask; default: 0.1"
        ),
    )

    parser.add_argument(
        "--density-ratio",
        "--density_ratio",
        dest="density_ratio",
        type=float,
        default=0.5,
        help="fraction of LiDAR points to retain, range [0, 1]",
    )

    parser.add_argument(
        "--drop-ratio",
        "--drop_ratio",
        dest="drop_ratio",
        type=float,
        default=0.5,
        help="fraction of LiDAR points to drop, range [0, 1]",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--map-threshold",
        type=float,
        default=0.5,
        help="fixed probability threshold used for all BEV metrics",
    )

    parser.add_argument(
        "--skip-camera-ablation",
        action="store_true",
        help="skip the camera-feature-zero control to reduce inference cost",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device, e.g. cuda:5; if omitted, use torchpack local_rank",
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings, key=value",
    )

    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default=None,
        help="optional dataset root override",
    )

    parser.add_argument(
        "--ann-file",
        "--ann_file",
        dest="ann_file",
        default=None,
        help="optional test annotation file override",
    )

    parser.add_argument(
        "--show-dir",
        "--show_dir",
        dest="show_dir",
        default=None,
        help="optional directory for saving BEV prediction/GT PNGs",
    )

    args = parser.parse_args()

    # 优先使用 --config / --checkpoint；否则使用位置参数
    args.config = args.config or args.config_pos
    args.checkpoint = args.checkpoint or args.checkpoint_pos

    if args.config is None:
        parser.error("config is required: use --config CONFIG or positional CONFIG")

    if args.checkpoint is None:
        parser.error(
            "checkpoint is required: use --checkpoint CHECKPOINT or positional CHECKPOINT"
        )

    if not 0.0 <= args.map_threshold <= 1.0:
        parser.error("--map-threshold must be in [0, 1]")

    if not 0.0 <= args.region_keep_ratio <= 1.0:
        parser.error("--region-keep-ratio must be in [0, 1]")

    # 清掉内部位置参数，避免后面误用
    del args.config_pos
    del args.checkpoint_pos

    return args


# =========================================================================
# 工具函数
# =========================================================================

def unwrap_dataset(dataset):
    """兼容 ConcatDataset 等包装。"""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def tensor_to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================================
# LiDAR 退化
# =========================================================================

class LidarDegrader:
    """
    只负责修改一个样本的 points Tensor。

    与原脚本最大的区别：
    不把退化后的点云重新包装成一个新的 DataContainer。
    而是保持原 DataContainer，只替换其中的 Tensor。

    这样可以最大程度保持项目原 DataLoader -> MMDataParallel -> model
    的数据结构不变。
    """

    def __init__(
        self,
        degradation,
        region=None,
        region_keep_ratio=0.1,
        density_ratio=0.5,
        drop_ratio=0.5,
        bev_scope=None,
    ):
        self.degradation = degradation
        self.region = dict(DEFAULT_REGION)
        if region is not None:
            self.region.update(region)
        self.region_keep_ratio = float(region_keep_ratio)
        self.density_ratio = float(density_ratio)
        self.drop_ratio = float(drop_ratio)
        self.bev_scope = bev_scope or (
            (0.0, 3.0, 0.02),
            (-1.5, 1.5, 0.02),
        )

        if not 0.0 <= self.region_keep_ratio <= 1.0:
            raise ValueError("--region-keep-ratio must be in [0, 1]")

        if not 0.0 <= self.density_ratio <= 1.0:
            raise ValueError("--density-ratio must be in [0, 1]")

        if not 0.0 <= self.drop_ratio <= 1.0:
            raise ValueError("--drop-ratio must be in [0, 1]")

    def full_remove(self, points):
        """Return a genuinely empty cloud; BEVFusion handles empty voxelization."""
        return points[:0].clone()

    def region_membership(self, points):
        x_min = float(self.region.get("x_min", 0.0))
        x_max = float(self.region.get("x_max", 3.04))
        y_min = float(self.region.get("y_min", -1.52))
        y_max = float(self.region.get("y_max", 1.52))
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid --region bounds: {self.region}")
        xyz = points[:, :3]
        return (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] <= y_max)
        )

    def count_region_points(self, points):
        if self.degradation != "region_mask":
            return None
        return int(self.region_membership(points).sum().item())

    def region_mask(self, points):
        """Randomly retain only part of the points inside the target region."""
        inside = self.region_membership(points)
        inside_indices = torch.nonzero(inside, as_tuple=False).flatten()
        num_keep = int(round(inside_indices.numel() * self.region_keep_ratio))

        keep_mask = ~inside
        if num_keep > 0:
            selected = inside_indices[
                torch.randperm(
                    inside_indices.numel(),
                    device=points.device,
                )[:num_keep]
            ]
            keep_mask[selected] = True
        return points[keep_mask]

    @staticmethod
    def random_keep(points, keep_ratio):
        keep_ratio = float(keep_ratio)
        if keep_ratio >= 1.0:
            return points
        if keep_ratio <= 0.0:
            return points[:0].clone()
        n = points.shape[0]
        if n == 0:
            return points
        n_keep = max(1, int(round(n * keep_ratio)))
        indices = torch.randperm(n, device=points.device)[:n_keep]
        return points[indices]

    def degrade(self, points):
        if self.degradation == "full_lidar_remove":
            return self.full_remove(points)
        if self.degradation == "region_mask":
            return self.region_mask(points)
        if self.degradation == "density_reduce":
            return self.random_keep(points, self.density_ratio)
        if self.degradation == "random_drop":
            return self.random_keep(points, 1.0 - self.drop_ratio)
        raise ValueError(f"Unsupported degradation: {self.degradation}")

    def missing_bev_mask(self, height, width):
        """Build the evaluated missing-region mask in label coordinates."""
        if self.degradation != "region_mask":
            return np.ones((height, width), dtype=bool)

        (bev_x_min, bev_x_max, _), (bev_y_min, bev_y_max, _) = self.bev_scope
        x = bev_x_min + (np.arange(height) + 0.5) * (
            (bev_x_max - bev_x_min) / height
        )
        y = bev_y_min + (np.arange(width) + 0.5) * (
            (bev_y_max - bev_y_min) / width
        )
        x_selected = (x >= float(self.region["x_min"])) & (
            x <= float(self.region["x_max"])
        )
        y_selected = (y >= float(self.region["y_min"])) & (
            y <= float(self.region["y_max"])
        )
        return x_selected[:, None] & y_selected[None, :]


# =========================================================================
# 保持原 DataLoader / DataContainer 结构
# =========================================================================

def get_points_from_data(data):
    if "points" not in data:
        raise KeyError(f"Current data does not contain key 'points'. Available keys: {list(data.keys())}")

    points = data["points"]

    # 1. 如果 points 是 DataContainer，取出其 .data
    if hasattr(points, "data"):
        points = points.data

    # 2. 如果 points 是 list/tuple，递归取出第一个元素（batch_size=1）
    while isinstance(points, (list, tuple)):
        if len(points) == 0:
            raise RuntimeError("Empty points container")
        points = points[0]

    # 3. 现在 points 应该是 numpy 数组或 torch.Tensor，如果不是 Tensor 则转换
    if not isinstance(points, torch.Tensor):
        # 若 points 是 numpy 数组或其他可转换类型，直接转换
        points = torch.as_tensor(points)

    return points


def replace_points_in_data(data, new_points):
    """
    深拷贝原 data，只替换 points 内部的 Tensor。

    关键点：
    - 不重新构造 DataContainer
    - 不改变 img / img_metas / lidar2img 等字段
    - 不改变原 DataLoader 产生的数据结构
    """
    new_data = copy.deepcopy(data)

    def replace_first_tensor(value):
        """Preserve DataContainer batch nesting while replacing one sample."""
        if isinstance(value, torch.Tensor):
            return new_points
        if isinstance(value, list):
            if not value:
                return [new_points]
            value[0] = replace_first_tensor(value[0])
            return value
        if isinstance(value, tuple):
            if not value:
                return (new_points,)
            return (replace_first_tensor(value[0]),) + value[1:]
        return new_points

    points_container = new_data["points"]

    if hasattr(points_container, "data"):
        points_data = points_container.data
        replaced_points = replace_first_tensor(points_data)
        if replaced_points is not points_data:
            # MMCV DataContainer.data is a read-only property backed by _data.
            points_container._data = replaced_points
    else:
        new_data["points"] = replace_first_tensor(points_container)

    return new_data


# =========================================================================
# BEV metrics
# =========================================================================

def metrics_from_counts(tp, fp, fn, valid_pixels, class_names):
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    valid_pixels = np.asarray(valid_pixels, dtype=np.float64)
    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    valid_classes = valid_pixels > 0

    per_class = {}
    for index, name in enumerate(class_names[: len(tp)]):
        per_class[name] = {
            "iou": float(iou[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "valid_pixels": int(valid_pixels[index]),
        }

    def valid_mean(values):
        if not np.any(valid_classes):
            return float("nan")
        return float(np.mean(values[valid_classes]))

    return {
        "mean_iou": valid_mean(iou),
        "mean_precision": valid_mean(precision),
        "mean_recall": valid_mean(recall),
        "mean_f1": valid_mean(f1),
        "per_class": per_class,
        "counts": {
            "tp": tp.astype(np.int64).tolist(),
            "fp": fp.astype(np.int64).tolist(),
            "fn": fn.astype(np.int64).tolist(),
            "valid_pixels": valid_pixels.astype(np.int64).tolist(),
        },
    }


def calculate_bev_metrics(
    pred_scores,
    gt_masks,
    class_names,
    supervision_mask=None,
    evaluation_mask=None,
    threshold=0.5,
):
    """Compute fixed-threshold confusion counts on a selected BEV region."""
    pred_scores = tensor_to_numpy(pred_scores)
    gt_masks = tensor_to_numpy(gt_masks).astype(bool)
    pred_bin = pred_scores >= float(threshold)

    if supervision_mask is not None:
        supervision_mask = tensor_to_numpy(supervision_mask).astype(bool)
    else:
        supervision_mask = np.ones_like(gt_masks, dtype=bool)

    num_classes = min(pred_bin.shape[0], gt_masks.shape[0])
    supervision_mask = supervision_mask[:num_classes]
    if evaluation_mask is not None:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask.ndim == 2:
            evaluation_mask = np.broadcast_to(
                evaluation_mask[None, :, :], supervision_mask.shape
            )
        supervision_mask = supervision_mask & evaluation_mask[:num_classes]

    tp, fp, fn, valid_pixels = [], [], [], []
    for c in range(num_classes):
        sup = supervision_mask[c]
        pred = pred_bin[c]
        gt = gt_masks[c]
        tp.append(int((np.logical_and(pred, gt) & sup).sum()))
        fp.append(int((np.logical_and(pred, ~gt) & sup).sum()))
        fn.append(int((np.logical_and(~pred, gt) & sup).sum()))
        valid_pixels.append(int(sup.sum()))

    return metrics_from_counts(tp, fp, fn, valid_pixels, class_names)


def draw_missing_region_box(image_path, missing_mask, label="LiDAR sparse"):
    """Draw the LiDAR-degraded BEV region directly on a saved PNG."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read BEV visualization: {image_path}")

    mask = np.asarray(missing_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return

    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    thickness = max(2, int(round(min(image.shape[:2]) / 75)))
    color = (0, 0, 255)  # OpenCV BGR: red
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(image.shape[:2]) / 500.0)
    text_size, baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_bottom = min(bottom, top + text_size[1] + baseline + 4)
    label_right = min(image.shape[1] - 1, left + text_size[0] + 6)
    cv2.rectangle(
        image,
        (left, top),
        (label_right, label_bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (left + 3, label_bottom - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(image_path), image)


# =========================================================================
# Tester
# =========================================================================

class LidarDegradationTester:

    def __init__(self, args):
        self.args = args

        # -------------------------------------------------------------
        # 1. 单 GPU 测试：不要调用 torchpack.dist.init()
        #
        # 原 tools/test.py 走 distributed test 流程，因此需要：
        #   dist.init()
        #   MASTER_HOST / MASTER_PORT / RANK ...
        #
        # 本脚本明确使用 --device cuda:N + dist=False 的单卡 DataLoader，
        # 因此这里直接设置 CUDA device，避免要求 MASTER_HOST。
        # -------------------------------------------------------------
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this test script.")

        if self.args.device is not None:
            if not self.args.device.startswith("cuda:"):
                raise ValueError(
                    f"--device must look like cuda:N, got: {self.args.device}"
                )
            gpu_id = int(self.args.device.split(":", 1)[1])
        else:
            gpu_id = torch.cuda.current_device()

        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid GPU id {gpu_id}; "
                f"available GPU count: {torch.cuda.device_count()}"
            )

        torch.cuda.set_device(gpu_id)

        self.gpu_id = gpu_id
        print(f"[Info] Using CUDA device: cuda:{self.gpu_id}")

        torch.backends.cudnn.benchmark = True

        # -------------------------------------------------------------
        # 2. 与 tools/test.py 完全一致的 Config 加载方式
        # -------------------------------------------------------------
        configs.load(args.config, recursive=True)
        configs.update(args.cfg_options or {})

        self.cfg = Config(
            recursive_eval(configs),
            filename=args.config,
        )

        print(f"[Info] Loaded config: {args.config}")

        if args.cfg_options:
            self.cfg.merge_from_dict(args.cfg_options)

        # -------------------------------------------------------------
        # 3. 数据路径覆盖
        # -------------------------------------------------------------
        self._override_dataset_paths()

        # -------------------------------------------------------------
        # 4. 与 tools/test.py 一致：test_mode + samples_per_gpu
        # -------------------------------------------------------------
        self.cfg.model.pretrained = None

        samples_per_gpu = 1

        if isinstance(self.cfg.data.test, dict):
            self.cfg.data.test.test_mode = True
            samples_per_gpu = self.cfg.data.test.pop(
                "samples_per_gpu",
                1,
            )

        elif isinstance(self.cfg.data.test, list):
            for ds_cfg in self.cfg.data.test:
                ds_cfg.test_mode = True

            samples_per_gpu = max(
                [
                    ds_cfg.pop("samples_per_gpu", 1)
                    for ds_cfg in self.cfg.data.test
                ]
            )

        # 为了让 points 处理最简单、最稳定，这里强制 batch=1。
        if samples_per_gpu != 1:
            print(
                f"[Warning] Config requested samples_per_gpu={samples_per_gpu}. "
                "Force to 1 for LiDAR degradation test."
            )
            samples_per_gpu = 1

        # -------------------------------------------------------------
        # 5. build dataset / dataloader
        # 与 tools/test.py: 303-310 保持一致
        # -------------------------------------------------------------
        self.dataset = build_dataset(self.cfg.data.test)
        base_dataset = unwrap_dataset(self.dataset)
        self.map_classes = list(
            getattr(base_dataset, "map_classes", None)
            or self.cfg.model.heads.map.classes
        )

        self.data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

        # -------------------------------------------------------------
        # 6. build model / load checkpoint
        # 与 tools/test.py: 313-336 保持一致
        # -------------------------------------------------------------
        self.cfg.model.train_cfg = None

        self.model = build_model(
            self.cfg.model,
            test_cfg=self.cfg.get("test_cfg"),
        )

        fp16_cfg = self.cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)

        checkpoint = load_checkpoint(
            self.model,
            args.checkpoint,
            map_location="cpu",
        )

        if "CLASSES" in checkpoint.get("meta", {}):
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            self.model.CLASSES = getattr(
                self.dataset,
                "CLASSES",
                None,
            )

        # 关键：复用原 test.py 的 MMDataParallel
        self.model = MMDataParallel(
            self.model.cuda(),
            device_ids=[self.gpu_id],
        )

        self.model.eval()
        self._print_cuda_diagnostics("after model load")
        self._lidar_diagnostics_printed = False
        if "lidar" in self.model.module.encoders:
            self.model.module.encoders["lidar"]["backbone"].register_forward_pre_hook(
                self._print_lidar_backbone_input_once
            )

        self.camera_ablation_enabled = not args.skip_camera_ablation
        if (
            self.camera_ablation_enabled
            and "camera" not in self.model.module.encoders
        ):
            warnings.warn(
                "Model has no camera encoder; skip camera-feature ablation."
            )
            self.camera_ablation_enabled = False

        # -------------------------------------------------------------
        # 7. LiDAR degradation
        # -------------------------------------------------------------
        region = None
        if args.region:
            region = json.loads(args.region)

        map_head = self.cfg.model.heads.get("map")
        bev_scope = None
        if map_head is not None and map_head.get("grid_transform") is not None:
            bev_scope = map_head.grid_transform.output_scope

        self.degrader = LidarDegrader(
            degradation=args.degradation,
            region=region,
            region_keep_ratio=args.region_keep_ratio,
            density_ratio=args.density_ratio,
            drop_ratio=args.drop_ratio,
            bev_scope=bev_scope,
        )

        # -------------------------------------------------------------
        # 8. 输出
        # -------------------------------------------------------------
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.show_dir = None
        if args.show_dir:
            self.show_dir = Path(args.show_dir)
            self.show_dir.mkdir(parents=True, exist_ok=True)

    def _print_cuda_diagnostics(self, stage):
        """Print the memory visible to this process, not host-wide nvidia-smi."""
        device = torch.device(f"cuda:{self.gpu_id}")
        properties = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        memory_parts = [
            f"allocated={allocated / 1024 ** 3:.2f} GiB",
            f"reserved={reserved / 1024 ** 3:.2f} GiB",
        ]
        try:
            free, total = torch.cuda.mem_get_info(device)
            memory_parts.extend(
                [
                    f"free={free / 1024 ** 3:.2f} GiB",
                    f"total={total / 1024 ** 3:.2f} GiB",
                ]
            )
        except (AttributeError, RuntimeError, TypeError):
            memory_parts.append(
                f"total={properties.total_memory / 1024 ** 3:.2f} GiB"
            )

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"[CUDA] {stage}: logical cuda:{self.gpu_id}, "
            f"name={properties.name}, CUDA_VISIBLE_DEVICES={visible_devices}, "
            + ", ".join(memory_parts)
        )

    def _print_lidar_backbone_input_once(self, _module, inputs):
        """Report the first voxelized input immediately before sparse conv."""
        if self._lidar_diagnostics_printed or len(inputs) < 2:
            return

        self._lidar_diagnostics_printed = True
        features, coordinates = inputs[:2]
        coordinate_range = "empty"
        if coordinates.numel() > 0:
            coord_min = coordinates.min(dim=0)[0].detach().cpu().tolist()
            coord_max = coordinates.max(dim=0)[0].detach().cpu().tolist()
            coordinate_range = f"min={coord_min}, max={coord_max}"

        print(
            f"[LiDAR] sparse input: voxels={features.shape[0]}, "
            f"feature_shape={tuple(features.shape)}, "
            f"coordinate_shape={tuple(coordinates.shape)}, "
            f"coordinate_range=({coordinate_range})"
        )
        self._print_cuda_diagnostics("before first sparse encoder")

    def _override_dataset_paths(self):
        """
        只在用户显式提供 --data-root / --ann-file 时覆盖。
        不主动修改原 config，避免改变正常推理脚本的行为。
        """
        if self.args.data_root is None and self.args.ann_file is None:
            return

        data_test = self.cfg.data.test

        if isinstance(data_test, list):
            data_cfgs = data_test
        else:
            data_cfgs = [data_test]

        for ds_cfg in data_cfgs:
            if self.args.data_root is not None and "dataset_root" in ds_cfg:
                ds_cfg.dataset_root = self.args.data_root
                if self.args.ann_file is None and "ann_file" in ds_cfg:
                    ds_cfg.ann_file = os.path.join(
                        self.args.data_root,
                        os.path.basename(ds_cfg.ann_file),
                    )

            if self.args.ann_file is not None:
                ds_cfg.ann_file = self.args.ann_file

        if self.args.data_root is not None:
            print(f"[Info] dataset_root override: {self.args.data_root}")

        if self.args.ann_file is not None:
            print(f"[Info] ann_file override: {self.args.ann_file}")

    # -----------------------------------------------------------------
    # 单次 forward
    # -----------------------------------------------------------------

    @torch.no_grad()
    def forward(self, data):
        """
        完全沿用正常 test.py 的调用形式：

            model(return_loss=False, rescale=True, **data)

        不直接调用 model.module，也不自己拼 BEVFusion 输入。
        """
        return self.model(
            return_loss=False,
            rescale=True,
            **data,
        )

    @torch.no_grad()
    def forward_with_camera_ablation(self, data):
        """Zero the camera BEV feature while preserving the trained graph."""

        def zero_tensors(value):
            if isinstance(value, torch.Tensor):
                return torch.zeros_like(value)
            if isinstance(value, tuple):
                return tuple(zero_tensors(item) for item in value)
            if isinstance(value, list):
                return [zero_tensors(item) for item in value]
            return value

        vtransform = self.model.module.encoders["camera"]["vtransform"]
        handle = vtransform.register_forward_hook(
            lambda _module, _inputs, output: zero_tensors(output)
        )
        try:
            return self.forward(data)
        finally:
            handle.remove()

    # -----------------------------------------------------------------
    # 单样本测试
    # -----------------------------------------------------------------

    def test_sample(self, data, sample_idx):
        original_points = get_points_from_data(data)

        if original_points.ndim != 2:
            raise RuntimeError(
                f"Unexpected points shape: {tuple(original_points.shape)}"
            )

        original_n = int(original_points.shape[0])
        original_region_n = self.degrader.count_region_points(original_points)
        torch.manual_seed(self.args.seed + sample_idx)

        # -------------------------------------------------------------
        # A. Original
        # -------------------------------------------------------------
        original_output = self.forward(data)

        # -------------------------------------------------------------
        # B. Camera only
        # -------------------------------------------------------------
        camera_points = self.degrader.full_remove(original_points)

        camera_data = replace_points_in_data(
            data,
            camera_points,
        )

        camera_output = self.forward(camera_data)

        # -------------------------------------------------------------
        # C. Degraded LiDAR
        # -------------------------------------------------------------
        degraded_points = self.degrader.degrade(original_points)

        degraded_data = replace_points_in_data(
            data,
            degraded_points,
        )

        degraded_output = self.forward(degraded_data)

        camera_ablated_output = None
        if self.camera_ablation_enabled:
            camera_ablated_output = self.forward_with_camera_ablation(
                degraded_data
            )

        # MMDataParallel + batch=1 -> output 通常为 list
        original_result = original_output[0]
        camera_result = camera_output[0]
        degraded_result = degraded_output[0]
        camera_ablated_result = (
            camera_ablated_output[0]
            if camera_ablated_output is not None
            else None
        )

        degraded_n = int(degraded_points.shape[0])
        degraded_region_n = self.degrader.count_region_points(degraded_points)
        reference_mask = original_result.get(
            "gt_masks_bev",
            original_result.get("masks_bev"),
        )
        if reference_mask is None:
            missing_bev_mask = None
        else:
            height, width = tensor_to_numpy(reference_mask).shape[-2:]
            missing_bev_mask = self.degrader.missing_bev_mask(height, width)

        result = {
            "original": original_result,
            "camera_only": camera_result,
            "degraded": degraded_result,
            "camera_ablated_degraded": camera_ablated_result,
            "missing_bev_mask": missing_bev_mask,
            "original_n_points": original_n,
            "degraded_n_points": degraded_n,
            "point_cloud_ratio": degraded_n / max(original_n, 1),
            "original_region_n_points": original_region_n,
            "degraded_region_n_points": degraded_region_n,
            "region_point_ratio": (
                degraded_region_n / max(original_region_n, 1)
                if original_region_n is not None
                else None
            ),
        }

        return result

    # -----------------------------------------------------------------
    # 指标
    # -----------------------------------------------------------------

    def compute_metrics(self, result):
        outputs = {
            "original": result["original"],
            "degraded": result["degraded"],
            "camera_only": result["camera_only"],
        }
        if result.get("camera_ablated_degraded") is not None:
            outputs["camera_ablated_degraded"] = result[
                "camera_ablated_degraded"
            ]

        if any("masks_bev" not in output for output in outputs.values()):
            return {"has_masks_bev": False}

        original = outputs["original"]
        if "gt_masks_bev" not in original:
            return {"has_masks_bev": True, "has_gt": False}

        gt = original["gt_masks_bev"]
        supervision = original.get("gt_supervision_mask_bev")
        missing_mask = result["missing_bev_mask"]
        region_masks = {
            "full": None,
            "missing": missing_mask,
            "retained": None if missing_mask is None else ~missing_mask,
        }

        regions = {}
        for region_name, region_mask in region_masks.items():
            regions[region_name] = {}
            for mode, output in outputs.items():
                regions[region_name][mode] = calculate_bev_metrics(
                    output["masks_bev"],
                    gt,
                    self.map_classes,
                    supervision_mask=supervision,
                    evaluation_mask=region_mask,
                    threshold=self.args.map_threshold,
                )

        return {
            "has_masks_bev": True,
            "has_gt": True,
            "regions": regions,
        }

    # -----------------------------------------------------------------
    # 可选保存 BEV prediction / GT PNG
    # -----------------------------------------------------------------

    def save_bev_visualizations(self, sample_idx, result):
        if self.show_dir is None:
            return

        sample_dir = self.show_dir / f"sample_{sample_idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        bev_png_paths = []

        for mode, output_name in [
            ("original", "original"),
            ("degraded", "sparse"),
            (
                "camera_ablated_degraded",
                "sparse_camera_ablated",
            ),
        ]:
            output = result.get(mode)

            if output is None or "masks_bev" not in output:
                continue

            masks = tensor_to_numpy(output["masks_bev"])
            png_path = sample_dir / f"{output_name}_masks_bev.png"
            visualize_map_scores(
                str(png_path),
                masks,
                classes=self.map_classes,
                threshold=self.args.map_threshold,
            )
            bev_png_paths.append(png_path)

        if "gt_masks_bev" in result["original"]:
            gt = tensor_to_numpy(result["original"]["gt_masks_bev"])
            gt_png_path = sample_dir / "gt_masks_bev.png"
            visualize_map(
                str(gt_png_path),
                gt.astype(bool),
                classes=self.map_classes,
            )
            bev_png_paths.append(gt_png_path)

        if result.get("missing_bev_mask") is not None:
            missing_mask = result["missing_bev_mask"].astype(np.uint8)
            for png_path in bev_png_paths:
                draw_missing_region_box(png_path, missing_mask)

    # -----------------------------------------------------------------
    # 主测试
    # -----------------------------------------------------------------

    def run(self):
        if self.args.num_samples <= 0:
            num_samples = len(self.data_loader)
        else:
            num_samples = min(self.args.num_samples, len(self.data_loader))

        print("\n" + "=" * 80)
        print("LiDAR Degradation Test")
        print("=" * 80)
        print(f"Config      : {self.args.config}")
        print(f"Checkpoint  : {self.args.checkpoint}")
        print(f"Degradation : {self.args.degradation}")
        print(f"Region      : {self.degrader.region}")
        if self.args.degradation == "region_mask":
            print(f"Region keep : {self.degrader.region_keep_ratio:.2%}")
        print(f"Threshold   : {self.args.map_threshold}")
        print(f"Camera ablate: {self.camera_ablation_enabled}")
        print(f"Num samples : {num_samples}")
        print(f"Output dir  : {self.output_dir}")
        print(f"Viz samples : {self.args.viz_samples}")
        print("=" * 80)

        sample_results = []

        start_time = time.perf_counter()

        for idx, data in enumerate(self.data_loader):
            if idx >= num_samples:
                break

            print(
                f"\n[{idx + 1}/{num_samples}] "
                f"processing sample {idx} ..."
            )

            try:
                result = self.test_sample(
                    data,
                    idx,
                )

                metrics = self.compute_metrics(result)

                metrics["original_n_points"] = result["original_n_points"]
                metrics["degraded_n_points"] = result["degraded_n_points"]
                metrics["point_cloud_ratio"] = result["point_cloud_ratio"]
                metrics["original_region_n_points"] = result[
                    "original_region_n_points"
                ]
                metrics["degraded_region_n_points"] = result[
                    "degraded_region_n_points"
                ]
                metrics["region_point_ratio"] = result["region_point_ratio"]

                sample_record = {
                    "index": idx,
                    **metrics,
                }

                sample_results.append(sample_record)

                print(
                    f"  LiDAR points: "
                    f"{result['original_n_points']} -> "
                    f"{result['degraded_n_points']} "
                    f"({result['point_cloud_ratio']:.2%})"
                )
                if result["original_region_n_points"] is not None:
                    print(
                        "  Region points: "
                        f"{result['original_region_n_points']} -> "
                        f"{result['degraded_region_n_points']} "
                        f"({result['region_point_ratio']:.2%})"
                    )

                if metrics.get("has_gt", False):
                    missing = metrics["regions"]["missing"]
                    iou_parts = [
                        f"Original={missing['original']['mean_iou']:.4f}",
                        f"Degraded={missing['degraded']['mean_iou']:.4f}",
                        f"CameraOnly={missing['camera_only']['mean_iou']:.4f}",
                    ]
                    if "camera_ablated_degraded" in missing:
                        iou_parts.append(
                            "CameraAblated="
                            f"{missing['camera_ablated_degraded']['mean_iou']:.4f}"
                        )
                    print("  Sparse-region IoU: " + ", ".join(iou_parts))
                else:
                    print("  GT masks not available; skip IoU.")

                if idx < self.args.viz_samples:
                    self.save_bev_visualizations(
                        idx,
                        result,
                    )

            except Exception as e:
                print(
                    f"  ERROR on sample {idx}: "
                    f"{type(e).__name__}: {e}"
                )

                import traceback
                traceback.print_exc()

                error_text = str(e).lower()
                is_cuda_error = isinstance(e, RuntimeError) and any(
                    marker in error_text
                    for marker in (
                        "cuda error",
                        "cuda execution failed",
                        "cuda runtime",
                        "cuda out of memory",
                        "cudnn error",
                        "cublas error",
                        "device-side assert",
                    )
                )
                if is_cuda_error:
                    print(
                        "[Fatal] CUDA errors can leave the process context "
                        "invalid; stop instead of retrying subsequent samples."
                    )
                    raise

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 数据等非 CUDA 的单样本异常仍允许继续批量测试。
                continue

        elapsed = time.perf_counter() - start_time

        summary = self.compute_summary(
            sample_results,
        )

        output = {
            "config": os.path.abspath(self.args.config),
            "checkpoint": os.path.abspath(self.args.checkpoint),
            "degradation": self.args.degradation,
            "degradation_config": {
                "region": self.degrader.region,
                "region_keep_ratio": self.degrader.region_keep_ratio,
                "density_ratio": self.degrader.density_ratio,
                "drop_ratio": self.degrader.drop_ratio,
            },
            "map_threshold": self.args.map_threshold,
            "camera_feature_ablation": self.camera_ablation_enabled,
            "num_samples_requested": num_samples,
            "num_samples_success": len(sample_results),
            "elapsed_seconds": elapsed,
            "summary": summary,
            "samples": sample_results,
        }

        output_path = self.output_dir / "results.json"

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print("\n" + "=" * 80)
        print("Test finished")
        print("=" * 80)
        print(f"Successful samples: {len(sample_results)}/{num_samples}")
        print(f"Elapsed            : {elapsed:.2f}s")
        print(f"Results            : {output_path}")

        if summary.get("regions"):
            print("\nAggregated sparse-region mIoU:")
            for mode, values in summary["regions"]["missing"].items():
                print(f"  {mode}: {values['mean_iou']:.6f}")
            if summary.get("comparisons"):
                print("Comparisons:")
                for key, value in summary["comparisons"].items():
                    print(f"  {key}: {value:+.6f}")

        return output

    def compute_summary(self, sample_results):
        if not sample_results:
            return {}

        summary = {
            "num_success": len(sample_results),
        }

        ratios = [
            x["point_cloud_ratio"]
            for x in sample_results
            if "point_cloud_ratio" in x
        ]

        if ratios:
            summary["avg_point_cloud_ratio"] = float(np.mean(ratios))

        region_ratios = [
            x["region_point_ratio"]
            for x in sample_results
            if x.get("region_point_ratio") is not None
        ]
        if region_ratios:
            summary["avg_region_point_ratio"] = float(np.mean(region_ratios))

        metric_results = [
            item
            for item in sample_results
            if item.get("has_gt", False) and item.get("regions")
        ]
        if not metric_results:
            return summary

        summary["regions"] = {}
        for region_name in metric_results[0]["regions"]:
            summary["regions"][region_name] = {}
            for mode in metric_results[0]["regions"][region_name]:
                counts = [
                    item["regions"][region_name][mode]["counts"]
                    for item in metric_results
                    if mode in item["regions"][region_name]
                ]
                if not counts:
                    continue
                tp = np.sum([value["tp"] for value in counts], axis=0)
                fp = np.sum([value["fp"] for value in counts], axis=0)
                fn = np.sum([value["fn"] for value in counts], axis=0)
                valid_pixels = np.sum(
                    [value["valid_pixels"] for value in counts], axis=0
                )
                summary["regions"][region_name][mode] = metrics_from_counts(
                    tp,
                    fp,
                    fn,
                    valid_pixels,
                    self.map_classes,
                )

        missing = summary["regions"].get("missing", {})
        comparisons = {}

        def add_difference(name, left, right):
            if left not in missing or right not in missing:
                return
            left_value = missing[left]["mean_iou"]
            right_value = missing[right]["mean_iou"]
            if np.isfinite(left_value) and np.isfinite(right_value):
                comparisons[name] = float(left_value - right_value)

        add_difference(
            "degraded_minus_original",
            "degraded",
            "original",
        )
        add_difference(
            "degraded_minus_camera_only",
            "degraded",
            "camera_only",
        )
        add_difference(
            "camera_supplement_gain",
            "degraded",
            "camera_ablated_degraded",
        )

        required = {"original", "degraded", "camera_ablated_degraded"}
        if required.issubset(missing):
            original_iou = missing["original"]["mean_iou"]
            degraded_iou = missing["degraded"]["mean_iou"]
            ablated_iou = missing["camera_ablated_degraded"]["mean_iou"]
            denominator = original_iou - ablated_iou
            if all(np.isfinite(v) for v in (original_iou, degraded_iou, ablated_iou)) and abs(denominator) > 1e-7:
                comparisons["recovery_ratio"] = float(
                    (degraded_iou - ablated_iou) / denominator
                )

        summary["comparisons"] = comparisons

        return summary


# =========================================================================
# main
# =========================================================================

def main():
    args = parse_args()

    if args.seed is not None:
        set_random_seed(
            args.seed,
            deterministic=False,
        )

    tester = LidarDegradationTester(args)
    tester.run()


if __name__ == "__main__":
    main()
