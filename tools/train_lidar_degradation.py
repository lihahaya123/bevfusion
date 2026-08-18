"""Train BEVFusion with realistic partial LiDAR degradation.

This entry point follows ``tools/train.py`` and changes only the training
pipeline. Validation/test data remain clean. For every training sample it
chooses one of four mutually exclusive modes:

* clean: keep the complete point cloud;
* local: remove one or two random x/y boxes across all z and loaded sweeps;
* global: randomly drop part of the complete point cloud;
* combined: apply a local box removal followed by global random dropping.

No mode removes the complete point cloud. BEV ground truth is left unchanged,
so the model is trained to predict complete BEV semantics from degraded LiDAR
and RGB.

Example::

torchpack dist-run -np 1 python tools/train_lidar_degradation.py \
  configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml \
  --run-dir data/replica_lidar_degradation_train \
  --finetune-from data/replica_base_train/best_robotbev_map_iou_max_epoch_2.pth \
  --clean-prob 0.30 \
  --local-prob 0.30 \
  --global-prob 0.20 \
  --combined-prob 0.20 \
  --local-size-x 0.5 1.0 \
  --local-size-y 0.5 1.0 \
  --global-drop-ratio 0.1 0.3 \
  --two-box-prob 0 \
  --min-points 128 \
  dataset_root=./data/replica_robot_bev_v4/ \
  max_epochs=10 \
  optimizer.lr=2.0e-5
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmcv import Config
from mmdet.datasets.builder import PIPELINES
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import (
    convert_sync_batchnorm,
    get_root_logger,
    load_checkpoint_selectively,
    recursive_eval,
)


@PIPELINES.register_module()
class RandomLidarDegradation:
    """Online local-region and global-density LiDAR degradation.

    The transform expects ``results['points']`` to be a BasePoints object. It
    should be inserted after multi-sweep loading, geometric augmentation and
    ``PointsRangeFilter`` so every loaded sweep is degraded in the final LiDAR
    coordinate system.
    """

    MODES = ("clean", "local", "global", "combined")

    def __init__(
        self,
        point_cloud_range,
        mode_probabilities=(0.3, 0.3, 0.2, 0.2),
        local_size_x=(0.5, 1.2),
        local_size_y=(0.5, 1.2),
        global_drop_ratio=(0.1, 0.4),
        two_box_prob=0.2,
        min_points=128,
    ):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        if self.point_cloud_range.shape != (6,):
            raise ValueError("point_cloud_range must contain 6 values")

        self.mode_probabilities = np.asarray(mode_probabilities, dtype=np.float64)
        if self.mode_probabilities.shape != (4,):
            raise ValueError("mode_probabilities must contain 4 values")
        if np.any(self.mode_probabilities < 0):
            raise ValueError("mode probabilities must be non-negative")
        if not np.isclose(self.mode_probabilities.sum(), 1.0):
            raise ValueError("mode probabilities must sum to 1")

        self.local_size_x = self._validate_range("local_size_x", local_size_x)
        self.local_size_y = self._validate_range("local_size_y", local_size_y)
        self.global_drop_ratio = self._validate_range(
            "global_drop_ratio", global_drop_ratio
        )
        if self.global_drop_ratio[0] < 0 or self.global_drop_ratio[1] >= 1:
            raise ValueError("global_drop_ratio must be within [0, 1)")
        if not 0 <= two_box_prob <= 1:
            raise ValueError("two_box_prob must be within [0, 1]")
        if min_points < 1:
            raise ValueError("min_points must be positive")

        x_extent = self.point_cloud_range[3] - self.point_cloud_range[0]
        y_extent = self.point_cloud_range[4] - self.point_cloud_range[1]
        if self.local_size_x[1] > x_extent or self.local_size_y[1] > y_extent:
            raise ValueError("local box size exceeds point-cloud range")

        self.two_box_prob = float(two_box_prob)
        self.min_points = int(min_points)

    @staticmethod
    def _validate_range(name, values):
        values = tuple(float(value) for value in values)
        if len(values) != 2 or values[0] < 0 or values[0] > values[1]:
            raise ValueError(f"{name} must be [minimum, maximum]")
        return values

    def _sample_mode(self):
        index = int(
            np.searchsorted(
                np.cumsum(self.mode_probabilities),
                np.random.random(),
                side="right",
            )
        )
        return self.MODES[min(index, len(self.MODES) - 1)]

    def _apply_local_boxes(self, point_tensor, keep_mask):
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        num_boxes = 2 if np.random.random() < self.two_box_prob else 1
        boxes = []

        for _ in range(num_boxes):
            size_x = np.random.uniform(*self.local_size_x)
            size_y = np.random.uniform(*self.local_size_y)
            if point_tensor.shape[0] > 0:
                # Anchor the random box on an observed point so a nominal
                # local-degradation sample does not accidentally mask empty BEV.
                anchor = point_tensor[np.random.randint(point_tensor.shape[0])]
                center_x = np.clip(
                    float(anchor[0]),
                    x_min + size_x / 2,
                    x_max - size_x / 2,
                )
                center_y = np.clip(
                    float(anchor[1]),
                    y_min + size_y / 2,
                    y_max - size_y / 2,
                )
            else:
                center_x = np.random.uniform(
                    x_min + size_x / 2, x_max - size_x / 2
                )
                center_y = np.random.uniform(
                    y_min + size_y / 2, y_max - size_y / 2
                )
            box = {
                "x_min": float(center_x - size_x / 2),
                "x_max": float(center_x + size_x / 2),
                "y_min": float(center_y - size_y / 2),
                "y_max": float(center_y + size_y / 2),
            }
            inside = (
                (point_tensor[:, 0] >= box["x_min"])
                & (point_tensor[:, 0] <= box["x_max"])
                & (point_tensor[:, 1] >= box["y_min"])
                & (point_tensor[:, 1] <= box["y_max"])
            )
            keep_mask &= ~inside
            boxes.append(box)

        return keep_mask, boxes

    def _apply_global_drop(self, keep_mask):
        available = torch.nonzero(keep_mask, as_tuple=False).flatten()
        drop_ratio = float(np.random.uniform(*self.global_drop_ratio))
        num_keep = int(round(available.numel() * (1.0 - drop_ratio)))
        num_keep = max(min(self.min_points, available.numel()), num_keep)

        selected = available[
            torch.randperm(available.numel(), device=available.device)[:num_keep]
        ]
        globally_kept = torch.zeros_like(keep_mask)
        globally_kept[selected] = True
        return globally_kept, drop_ratio

    def __call__(self, results):
        if "points" not in results:
            raise KeyError("RandomLidarDegradation requires results['points']")

        points = results["points"]
        point_tensor = points.tensor
        original_count = len(points)
        mode = self._sample_mode()
        keep_mask = torch.ones(
            original_count,
            dtype=torch.bool,
            device=point_tensor.device,
        )
        boxes = []
        drop_ratio = 0.0

        if mode in ("local", "combined"):
            keep_mask, boxes = self._apply_local_boxes(point_tensor, keep_mask)
        if mode in ("global", "combined"):
            keep_mask, drop_ratio = self._apply_global_drop(keep_mask)

        minimum = min(self.min_points, original_count)
        if int(keep_mask.sum()) < minimum:
            # Never turn an unusually sparse frame into an empty-LiDAR sample.
            keep_mask.fill_(True)
            mode = "clean_fallback"
            boxes = []
            drop_ratio = 0.0

        results["points"] = points[keep_mask]
        for key in ("pts_instance_mask", "pts_semantic_mask"):
            if key in results and len(results[key]) == original_count:
                mask = keep_mask.detach().cpu().numpy()
                results[key] = results[key][mask]

        results["lidar_degradation"] = {
            "mode": mode,
            "boxes": boxes,
            "global_drop_ratio": drop_ratio,
            "original_points": original_count,
            "retained_points": int(keep_mask.sum()),
        }
        return results

    def __repr__(self):
        probabilities = dict(zip(self.MODES, self.mode_probabilities.tolist()))
        return (
            f"{self.__class__.__name__}(probabilities={probabilities}, "
            f"local_size_x={self.local_size_x}, "
            f"local_size_y={self.local_size_y}, "
            f"global_drop_ratio={self.global_drop_ratio}, "
            f"two_box_prob={self.two_box_prob}, min_points={self.min_points})"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BEVFusion with partial LiDAR degradation"
    )
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--finetune-from",
        default=None,
        help="optional checkpoint used to initialize this robustness training",
    )
    parser.add_argument("--clean-prob", type=float, default=0.30)
    parser.add_argument("--local-prob", type=float, default=0.30)
    parser.add_argument("--global-prob", type=float, default=0.20)
    parser.add_argument("--combined-prob", type=float, default=0.20)
    parser.add_argument(
        "--local-size-x",
        type=float,
        nargs=2,
        default=(0.5, 1.2),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--local-size-y",
        type=float,
        nargs=2,
        default=(0.5, 1.2),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--global-drop-ratio",
        type=float,
        nargs=2,
        default=(0.1, 0.4),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--two-box-prob", type=float, default=0.20)
    parser.add_argument("--min-points", type=int, default=128)
    return parser.parse_known_args()


def insert_degradation_transform(cfg, args):
    probabilities = (
        args.clean_prob,
        args.local_prob,
        args.global_prob,
        args.combined_prob,
    )
    if any(probability < 0 for probability in probabilities) or not np.isclose(
        sum(probabilities), 1.0
    ):
        raise ValueError("clean/local/global/combined probabilities must sum to 1")

    transform_cfg = dict(
        type="RandomLidarDegradation",
        point_cloud_range=list(cfg.point_cloud_range),
        mode_probabilities=probabilities,
        local_size_x=tuple(args.local_size_x),
        local_size_y=tuple(args.local_size_y),
        global_drop_ratio=tuple(args.global_drop_ratio),
        two_box_prob=args.two_box_prob,
        min_points=args.min_points,
    )

    pipeline = cfg.data.train.pipeline
    insert_at = None
    for index, transform in enumerate(pipeline):
        if transform.get("type") == "RandomLidarDegradation":
            raise ValueError("training pipeline already contains RandomLidarDegradation")
        if transform.get("type") == "PointsRangeFilter":
            insert_at = index + 1
    if insert_at is None:
        raise ValueError("PointsRangeFilter not found in cfg.data.train.pipeline")

    pipeline.insert(insert_at, transform_cfg)
    cfg.lidar_degradation_training = transform_cfg


def main():
    dist.init()
    args, config_overrides = parse_args()

    configs.load(args.config, recursive=True)
    configs.update(config_overrides)
    cfg = Config(recursive_eval(configs), filename=args.config)

    insert_degradation_transform(cfg, args)
    if args.finetune_from is not None:
        cfg.load_from = args.finetune_from
        cfg.resume_from = None
        # This option targets a checkpoint from the same model, so load every
        # layer instead of applying the base config's cross-model skip list.
        cfg.load_from_ignore_shape_mismatch = False
        cfg.load_from_skip_prefixes = []

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(
        "LiDAR degradation training: "
        f"{cfg.lidar_degradation_training}"
    )

    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]
    model = build_model(cfg.model)
    model.init_weights()
    if (
        cfg.load_from
        and not cfg.resume_from
        and cfg.get("load_from_ignore_shape_mismatch", False)
    ):
        load_checkpoint_selectively(
            model,
            cfg.load_from,
            skip_prefixes=cfg.get("load_from_skip_prefixes", []),
            logger=logger,
        )
        cfg.load_from = None
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
