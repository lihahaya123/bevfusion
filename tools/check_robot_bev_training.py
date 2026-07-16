import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import load_checkpoint_selectively, recursive_eval
from torchpack.utils.config import configs


DEFAULT_CONFIG = "configs/robot_bev/seg/robotbev_camera_lidar_lss.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check RobotBEV data/config/model wiring before launching full training."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"training config file, default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="dataset split to check",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=0,
        help="workers used by the smoke dataloader",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device used for one-batch forward/backward",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="only build dataset/dataloader/model; do not run a CUDA batch",
    )
    args, opts = parser.parse_known_args()
    return args, opts


def load_config(config_path: str, opts) -> Config:
    configs.load(config_path, recursive=True)
    configs.update(opts)
    return Config(recursive_eval(configs), filename=config_path)


def unwrap_data_container(value: Any) -> Any:
    if isinstance(value, DataContainer):
        return value.data
    return value


def describe_value(value: Any) -> str:
    value = unwrap_data_container(value)
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, list):
        if not value:
            return "list[0]"
        return f"list[{len(value)}]({describe_value(value[0])})"
    if isinstance(value, tuple):
        if not value:
            return "tuple[0]"
        return f"tuple[{len(value)}]({describe_value(value[0])})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    return type(value).__name__


def main():
    args, opts = parse_args()
    cfg = load_config(args.config, opts)
    cfg.data.workers_per_gpu = args.workers_per_gpu

    dataset = build_dataset(cfg.data[args.split])
    print(f"[dataset] {dataset.__class__.__name__} split={args.split} len={len(dataset)}")
    print(f"[dataset] map_classes={getattr(dataset, 'map_classes', None)}")

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    batch = next(iter(data_loader))
    print(f"[dataloader] keys={list(batch.keys())}")
    for key in (
        "img",
        "points",
        "gt_masks_bev",
        "gt_supervision_mask_bev",
        "gt_bboxes_3d",
        "gt_labels_3d",
    ):
        if key in batch:
            print(f"[dataloader] {key}: {describe_value(batch[key])}")

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.load_from and cfg.get("load_from_ignore_shape_mismatch", False):
        load_checkpoint_selectively(
            model,
            cfg.load_from,
            skip_prefixes=cfg.get("load_from_skip_prefixes", []),
        )
    print(f"[model] {model.__class__.__name__} heads={list(model.heads.keys())}")

    if args.skip_forward:
        print("[forward] skipped by --skip-forward")
        return
    if not torch.cuda.is_available():
        print("[forward] skipped because CUDA is not available")
        return

    torch.cuda.set_device(args.gpu_id)
    model = MMDataParallel(model.cuda(), device_ids=[args.gpu_id])
    model.train()

    outputs = model.train_step(batch, optimizer=None)
    loss = outputs["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.detach().cpu().item()}")
    loss.backward()

    print(f"[forward] loss={loss.detach().cpu().item():.6f}")
    for name, value in outputs["log_vars"].items():
        print(f"[forward] {name}={value}")
    print("[forward] one-batch forward/backward passed")


if __name__ == "__main__":
    main()
