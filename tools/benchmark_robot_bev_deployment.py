"""Benchmark RobotBEV inference without accumulating dataset outputs.

This tool reports two deliberately separate scopes:

1. ``model``: a pre-scattered GPU batch is reused and only the public model
   inference call is timed.
2. ``end-to-end``: preprocessed CPU batches are reused and the timed region
   includes CPU-to-GPU scatter, model inference, and the model's result copy
   back to CPU.

Dataset file I/O, sensor acquisition, and dataset pipeline transforms happen
while batches are preloaded and are therefore excluded from both scopes.
"""

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet3d.utils.baseline_metrics import (
    model_statistics,
    runtime_environment,
    summarize_samples,
    write_json,
)


SUPERVISION_KEYS = {
    "depths",
    "gt_bboxes_3d",
    "gt_depths",
    "gt_labels_3d",
    "gt_masks_bev",
    "gt_supervision_mask_bev",
}


def parse_args() -> Tuple[argparse.Namespace, Sequence[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RobotBEV model-only and deployment-style inference "
            "without timing sensor acquisition, disk I/O, or dataset transforms."
        )
    )
    parser.add_argument("config", help="test config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--mode",
        choices=("both", "model", "end-to-end"),
        default="both",
        help="benchmark scope, default: both",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="dataset split used to preload representative batches",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="samples per inference call, default: 1",
    )
    parser.add_argument(
        "--preload-batches",
        type=int,
        default=16,
        help=(
            "number of preprocessed CPU batches retained in RAM for the "
            "end-to-end benchmark, default: 16"
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=0,
        help="workers used only while preloading batches, default: 0",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="untimed warmup iterations per scope, default: 50",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="timed iterations per scope, default: 500",
    )
    parser.add_argument(
        "--precision",
        choices=("config", "fp16", "fp32"),
        default="config",
        help="inference precision; config follows the fp16 config entry",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--keep-supervision",
        action="store_true",
        help=(
            "keep GT tensors in model inputs; disabled by default because "
            "deployment has no labels"
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="progress logging interval; 0 disables progress output",
    )
    parser.add_argument(
        "--output",
        help=(
            "JSON output path; default is deployment_benchmark_<timestamp>.json "
            "beside the checkpoint"
        ),
    )
    args, opts = parser.parse_known_args()
    return args, opts


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "preload_batches": args.preload_batches,
        "iterations": args.iterations,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.workers_per_gpu < 0:
        raise ValueError("--workers-per-gpu must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def load_config(config_path: str, opts: Sequence[str]) -> Config:
    configs.load(config_path, recursive=True)
    configs.update(opts)
    return Config(recursive_eval(configs), filename=config_path)


def disable_pretrained_initializers(value: Any) -> None:
    """Avoid loading backbone pretraining before a full checkpoint is loaded."""
    if isinstance(value, MutableMapping):
        for key, child in list(value.items()):
            if key == "pretrained":
                value[key] = None
            elif (
                key == "init_cfg"
                and isinstance(child, Mapping)
                and child.get("type") == "Pretrained"
            ):
                value[key] = None
            else:
                disable_pretrained_initializers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            disable_pretrained_initializers(child)


def strip_supervision(batch: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in batch.items() if key not in SUPERVISION_KEYS}


def build_cpu_batches(
    cfg: Config,
    split: str,
    batch_size: int,
    workers_per_gpu: int,
    preload_batches: int,
    keep_supervision: bool,
) -> Tuple[Any, List[Dict[str, Any]]]:
    data_cfg = copy.deepcopy(cfg.data[split])
    if not isinstance(data_cfg, MutableMapping):
        raise TypeError(
            "deployment benchmark currently expects one dataset config per split"
        )
    data_cfg["test_mode"] = True
    data_cfg.pop("samples_per_gpu", None)

    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    batches: List[Dict[str, Any]] = []
    for batch in data_loader:
        prepared = dict(batch) if keep_supervision else strip_supervision(batch)
        batches.append(prepared)
        if len(batches) >= preload_batches:
            break

    if not batches:
        raise RuntimeError(f"dataset split {split!r} produced no batches")
    return dataset, batches


def resolve_fp16(cfg: Config, precision: str) -> bool:
    if precision == "fp16":
        return True
    if precision == "fp32":
        return False
    return cfg.get("fp16", None) is not None


def build_inference_model(
    cfg: Config,
    checkpoint_path: str,
    gpu_id: int,
    use_fp16: bool,
) -> MMDataParallel:
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg["train_cfg"] = None
    disable_pretrained_initializers(model_cfg)

    model = build_model(model_cfg, test_cfg=cfg.get("test_cfg"))
    if use_fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    device = torch.device("cuda", gpu_id)
    model = model.to(device)
    model.eval()
    return MMDataParallel(model, device_ids=[gpu_id])


def settle_cuda(device: torch.device, release_cache: bool = False) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    if release_cache:
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def current_memory(device: torch.device) -> Dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "device_used_mb": (total_bytes - free_bytes) / (1024**2),
        "device_free_mb": free_bytes / (1024**2),
        "device_total_mb": total_bytes / (1024**2),
    }


def peak_memory(device: torch.device, idle: Mapping[str, float]) -> Dict[str, float]:
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    after_run = current_memory(device)
    return {
        "idle_allocated_mb": idle["allocated_mb"],
        "idle_reserved_mb": idle["reserved_mb"],
        "idle_device_used_mb": idle["device_used_mb"],
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "forward_increment_allocated_mb": max(
            peak_allocated - idle["allocated_mb"], 0.0
        ),
        "reserved_growth_mb": max(peak_reserved - idle["reserved_mb"], 0.0),
        "after_run_allocated_mb": after_run["allocated_mb"],
        "after_run_reserved_mb": after_run["reserved_mb"],
        "after_run_device_used_mb": after_run["device_used_mb"],
    }


def output_sample_count(output: Any, fallback: int) -> int:
    if isinstance(output, (list, tuple)):
        return len(output)
    return fallback


def log_progress(scope: str, completed: int, total: int, interval: int) -> None:
    if interval and (completed % interval == 0 or completed == total):
        print(f"[{scope}] {completed}/{total} iterations")


def finalize_result(
    scope: str,
    timer: str,
    times_ms: List[float],
    samples: int,
    memory: Mapping[str, float],
    warmup: int,
    iterations: int,
) -> Dict[str, Any]:
    latency = summarize_samples(times_ms)
    latency.update({"unit": "ms_per_batch", "timer": timer})
    total_seconds = sum(times_ms) / 1000.0
    return {
        "scope": scope,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "processed_samples": samples,
        "latency": latency,
        "performance": {
            "frames_per_second": samples / max(total_seconds, 1e-12),
            "batches_per_second": iterations / max(total_seconds, 1e-12),
            "timed_seconds": total_seconds,
        },
        "memory": dict(memory),
    }


def benchmark_model_only(
    parallel_model: MMDataParallel,
    gpu_batch: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
    log_interval: int,
) -> Dict[str, Any]:
    module = parallel_model.module

    with torch.inference_mode():
        for _ in range(warmup):
            output = module(return_loss=False, rescale=True, **gpu_batch)
            del output

    # Keep the allocator cache created by warmup so latency represents a
    # steady-state service. Resetting peak stats still isolates timed forwards.
    settle_cuda(device)
    idle = current_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    times_ms: List[float] = []
    samples = 0
    with torch.inference_mode():
        for index in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = module(return_loss=False, rescale=True, **gpu_batch)
            end.record()
            torch.cuda.synchronize(device)

            times_ms.append(float(start.elapsed_time(end)))
            samples += output_sample_count(output, batch_size)
            del output
            log_progress("model", index + 1, iterations, log_interval)

    memory = peak_memory(device, idle)
    return finalize_result(
        scope=(
            "pre-scattered GPU batch -> public model inference -> CPU result; "
            "excludes CPU-to-GPU input transfer"
        ),
        timer="CUDA event",
        times_ms=times_ms,
        samples=samples,
        memory=memory,
        warmup=warmup,
        iterations=iterations,
    )


def benchmark_end_to_end(
    parallel_model: MMDataParallel,
    cpu_batches: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
    log_interval: int,
) -> Dict[str, Any]:
    with torch.inference_mode():
        for index in range(warmup):
            batch = cpu_batches[index % len(cpu_batches)]
            output = parallel_model(return_loss=False, rescale=True, **batch)
            del output

    # Keep the allocator cache created by warmup so latency represents a
    # steady-state service. Resetting peak stats still isolates timed forwards.
    settle_cuda(device)
    idle = current_memory(device)
    torch.cuda.reset_peak_memory_stats(device)

    times_ms: List[float] = []
    samples = 0
    with torch.inference_mode():
        for index in range(iterations):
            batch = cpu_batches[index % len(cpu_batches)]
            torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            output = parallel_model(return_loss=False, rescale=True, **batch)
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - started_at) * 1000.0)

            samples += output_sample_count(output, batch_size)
            del output
            log_progress("end-to-end", index + 1, iterations, log_interval)

    memory = peak_memory(device, idle)
    return finalize_result(
        scope=(
            "preprocessed CPU batch -> GPU scatter -> model inference -> CPU result; "
            "excludes acquisition, disk I/O, and dataset transforms"
        ),
        timer="synchronized wall clock",
        times_ms=times_ms,
        samples=samples,
        memory=memory,
        warmup=warmup,
        iterations=iterations,
    )


def scatter_one_batch(
    parallel_model: MMDataParallel,
    cpu_batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    _, scattered_kwargs = parallel_model.scatter(
        (), dict(cpu_batch), parallel_model.device_ids
    )
    if len(scattered_kwargs) != 1:
        raise RuntimeError(
            "model-only benchmark requires exactly one target CUDA device"
        )
    return scattered_kwargs[0]


def default_output_path(checkpoint_path: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    checkpoint = Path(checkpoint_path).expanduser()
    return str(checkpoint.parent / f"deployment_benchmark_{timestamp}.json")


def print_result(name: str, result: Mapping[str, Any]) -> None:
    latency = result["latency"]
    performance = result["performance"]
    memory = result["memory"]
    print(f"\n{name}")
    print(
        "  latency: "
        f"mean={latency['mean']:.3f} ms, "
        f"p50={latency['p50']:.3f} ms, "
        f"p95={latency['p95']:.3f} ms, "
        f"p99={latency['p99']:.3f} ms"
    )
    print(
        "  throughput: "
        f"{performance['frames_per_second']:.3f} frames/s, "
        f"{performance['batches_per_second']:.3f} batches/s"
    )
    print(
        "  memory: "
        f"idle={memory['idle_allocated_mb']:.1f} MB, "
        f"forward_increment={memory['forward_increment_allocated_mb']:.1f} MB, "
        f"peak_allocated={memory['peak_allocated_mb']:.1f} MB, "
        f"peak_reserved={memory['peak_reserved_mb']:.1f} MB, "
        f"device_used_after_run={memory['after_run_device_used_mb']:.1f} MB"
    )


def main() -> None:
    args, opts = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deployment benchmarking")

    cfg = load_config(args.config, opts)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))

    use_fp16 = resolve_fp16(cfg, args.precision)
    precision = "fp16" if use_fp16 else "fp32"
    print(
        f"[setup] split={args.split} batch_size={args.batch_size} "
        f"precision={precision}"
    )
    print(
        "[setup] preloading preprocessed CPU batches; this stage is not timed"
    )
    dataset, cpu_batches = build_cpu_batches(
        cfg=cfg,
        split=args.split,
        batch_size=args.batch_size,
        workers_per_gpu=args.workers_per_gpu,
        preload_batches=args.preload_batches,
        keep_supervision=args.keep_supervision,
    )
    print(
        f"[setup] preloaded {len(cpu_batches)} batches from "
        f"{dataset.__class__.__name__}"
    )

    parallel_model = build_inference_model(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        gpu_id=args.gpu_id,
        use_fp16=use_fp16,
    )
    settle_cuda(device, release_cache=True)
    static_memory = current_memory(device)
    print(
        "[setup] model static CUDA tensors: "
        f"allocated={static_memory['allocated_mb']:.1f} MB, "
        f"reserved={static_memory['reserved_mb']:.1f} MB, "
        f"device_used={static_memory['device_used_mb']:.1f} MB"
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "environment": runtime_environment(),
        "model": model_statistics(parallel_model),
        "settings": {
            "mode": args.mode,
            "split": args.split,
            "dataset": dataset.__class__.__name__,
            "dataset_samples": len(dataset),
            "batch_size": args.batch_size,
            "preloaded_batches": len(cpu_batches),
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "precision": precision,
            "gpu_id": args.gpu_id,
            "supervision_included": args.keep_supervision,
            "single_stream_synchronous": True,
        },
        "static_memory": static_memory,
    }

    gpu_batch = None
    if args.mode in {"both", "model"}:
        gpu_batch = scatter_one_batch(parallel_model, cpu_batches[0])
        report["model_only"] = benchmark_model_only(
            parallel_model=parallel_model,
            gpu_batch=gpu_batch,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            log_interval=args.log_interval,
        )
        print_result("Model-only benchmark", report["model_only"])

    if gpu_batch is not None:
        del gpu_batch
        settle_cuda(device, release_cache=True)

    if args.mode in {"both", "end-to-end"}:
        report["end_to_end"] = benchmark_end_to_end(
            parallel_model=parallel_model,
            cpu_batches=cpu_batches,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            log_interval=args.log_interval,
        )
        print_result("End-to-end deployment benchmark", report["end_to_end"])

    output_path = args.output or default_output_path(args.checkpoint)
    write_json(output_path, report)
    print(f"\n[done] report written to {output_path}")


if __name__ == "__main__":
    main()
