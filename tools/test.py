import argparse
import copy
import os
import re
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmcv
import numpy as np
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core.utils import visualize_map, visualize_map_scores
from mmdet3d.utils import recursive_eval
from mmdet3d.utils.baseline_metrics import (
    InferenceLatencyRecorder,
    cuda_memory,
    model_statistics,
    process_memory,
    runtime_environment,
    write_json,
)


def is_robotbev_dataset(dataset) -> bool:
    if dataset.__class__.__name__ == "RobotBEVDataset":
        return True
    inner = getattr(dataset, "dataset", None)
    return inner is not None and is_robotbev_dataset(inner)


def unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def safe_visualization_name(value) -> str:
    name = str(value)
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._")
    return name or "unnamed"


def default_metrics_out_path(args, timestamp=None) -> str:
    checkpoint_name = safe_visualization_name(Path(args.checkpoint).stem)
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"metrics_{checkpoint_name}_{timestamp}.json"
    if args.show_dir:
        base_dir = Path(args.show_dir).expanduser().parent
    elif args.out:
        base_dir = Path(args.out).expanduser().parent
    else:
        base_dir = Path(args.checkpoint).expanduser().parent
    return str(base_dir / filename)


def default_baseline_out_path(args, timestamp) -> str:
    checkpoint_name = safe_visualization_name(Path(args.checkpoint).stem)
    filename = f"baseline_test_{checkpoint_name}_{timestamp}.json"
    if args.show_dir:
        base_dir = Path(args.show_dir).expanduser().parent
    elif args.out:
        base_dir = Path(args.out).expanduser().parent
    else:
        base_dir = Path(args.checkpoint).expanduser().parent
    return str(base_dir / filename)


def distributed_max(value):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.item()


def save_robotbev_visualizations(dataset, outputs, out_dir, map_score=0.5) -> None:
    dataset = unwrap_dataset(dataset)
    mmcv.mkdir_or_exist(out_dir)
    map_classes = getattr(dataset, "map_classes", None) or []

    for index, result in enumerate(outputs):
        token = dataset.data_infos[index].get("token", f"{index:06d}")
        name = safe_visualization_name(token)
        pred = result.get("masks_bev")
        gt = result.get("gt_masks_bev")
        if pred is not None:
            visualize_map_scores(
                os.path.join(out_dir, "map_pred", f"{name}.png"),
                pred.numpy(),
                classes=map_classes,
                threshold=map_score,
            )
        if gt is not None:
            gt_mask = gt.numpy().astype(bool)
            visualize_map(
                os.path.join(out_dir, "map_gt", f"{name}.png"),
                gt_mask,
                classes=map_classes,
            )
        if pred is not None and gt is not None:
            pred_any = (pred.numpy() > map_score).any(axis=0)
            gt_any = gt.numpy().astype(bool).any(axis=0)
            overlay = np.zeros((*pred_any.shape, 3), dtype=np.uint8)
            overlay[gt_any] = (0, 180, 0)
            overlay[pred_any] = (220, 0, 0)
            overlay[pred_any & gt_any] = (240, 220, 0)
            fpath = os.path.join(out_dir, "map_overlay", f"{name}.png")
            mmcv.mkdir_or_exist(os.path.dirname(fpath))
            mmcv.imwrite(overlay[:, :, ::-1], fpath)


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--metrics-out",
        help="path to save evaluation metrics, e.g. results/metrics.json",
    )
    parser.add_argument(
        "--baseline-metrics-out",
        help="path to save accuracy, latency, memory and model baseline metrics",
    )
    parser.add_argument(
        "--map-score",
        type=float,
        default=0.5,
        help="score threshold for saving RobotBEV map predictions with --show-dir",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args, opts = parser.parse_known_args()
    args.cfg_opts = opts
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    dist.init()
    run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if not (
        args.out
        or args.eval
        or args.format_only
        or args.show
        or args.show_dir
        or args.metrics_out
    ):
        args.eval = ["map"]
    if args.metrics_out and not args.eval:
        args.eval = ["map"]

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    configs.update(args.cfg_opts)
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    baseline_cfg = cfg.get("baseline_metrics", {})
    baseline_enabled = baseline_cfg.get("enabled", False)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
    else:
        model = MMDataParallel(model, device_ids=[0])

    latency_recorder = None
    if baseline_enabled:
        latency_recorder = InferenceLatencyRecorder(
            model,
            warmup=baseline_cfg.get("inference_warmup_batches", 5),
        ).start()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        inference_started_at = time.perf_counter()

    if distributed:
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    else:
        outputs = single_gpu_test(model, data_loader)

    if baseline_enabled:
        torch.cuda.synchronize()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        inference_seconds = distributed_max(time.perf_counter() - inference_started_at)
        latency_recorder.stop()
        latency = latency_recorder.summary()
        test_cuda_memory = cuda_memory()
        test_cuda_memory["peak_allocated_mb_max"] = distributed_max(
            test_cuda_memory.get("peak_allocated_mb", 0.0)
        )
        test_cuda_memory["peak_reserved_mb_max"] = distributed_max(
            test_cuda_memory.get("peak_reserved_mb", 0.0)
        )

    rank, world_size = get_dist_info()
    if rank == 0:
        metrics = {}
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        if args.show_dir and is_robotbev_dataset(dataset):
            print(f"\nwriting RobotBEV visualizations to {args.show_dir}")
            save_robotbev_visualizations(
                dataset, outputs, args.show_dir, map_score=args.map_score
            )
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            if is_robotbev_dataset(dataset):
                eval_kwargs.update(kwargs)
            else:
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metrics = dataset.evaluate(outputs, **eval_kwargs)
            print(metrics)
            metrics_out = args.metrics_out or default_metrics_out_path(
                args, run_timestamp
            )
            metrics_dir = os.path.dirname(metrics_out)
            if metrics_dir:
                mmcv.mkdir_or_exist(metrics_dir)
            print(f"\nwriting metrics to {metrics_out}")
            mmcv.dump(metrics, metrics_out)
        if baseline_enabled:
            sample_count = len(dataset)
            checkpoint_bytes = os.path.getsize(args.checkpoint)
            baseline = {
                "schema_version": 1,
                "phase": "test",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config": os.path.abspath(args.config),
                "checkpoint": {
                    "path": os.path.abspath(args.checkpoint),
                    "size_mb": checkpoint_bytes / (1024 * 1024),
                },
                "dataset_samples": sample_count,
                "world_size": world_size,
                "model": model_statistics(model),
                "environment": runtime_environment(),
                "performance": {
                    "batch_size_per_gpu": samples_per_gpu,
                    "end_to_end_seconds": inference_seconds,
                    "end_to_end_ms_per_sample": (
                        inference_seconds * 1000 / max(sample_count, 1)
                    ),
                    "end_to_end_samples_per_second": (
                        sample_count / max(inference_seconds, 1e-12)
                    ),
                    "model_latency": latency,
                },
                "memory": {
                    "process": process_memory(),
                    "cuda": test_cuda_memory,
                },
                "accuracy": metrics,
            }
            baseline_out = args.baseline_metrics_out or default_baseline_out_path(
                args, run_timestamp
            )
            write_json(baseline_out, baseline)
            print(
                "\nBaseline: "
                f"{baseline['performance']['end_to_end_samples_per_second']:.3f} "
                "samples/s, "
                f"{test_cuda_memory['peak_allocated_mb_max']:.1f} MB peak CUDA"
            )
            print(f"writing baseline metrics to {baseline_out}")


if __name__ == "__main__":
    main()
