import os
import time
from numbers import Number

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info

from mmdet3d.utils.baseline_metrics import (
    append_jsonl,
    cuda_memory,
    model_statistics,
    process_memory,
    runtime_environment,
)


def _distributed_value(value, operation):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=operation)
    return tensor.item()


@HOOKS.register_module()
class BaselineMetricsHook(Hook):
    """Write one JSONL baseline record for every training epoch."""

    def __init__(self, output_file="baseline_train_metrics.jsonl"):
        self.output_file = output_file
        self.output_path = None
        self.epoch_started_at = None
        self.epoch_elapsed = None
        self.train_peak_allocated = 0.0
        self.train_peak_reserved = 0.0
        self.local_samples = 0
        self.loss_sums = {}
        self.loss_weights = {}

    def before_run(self, runner):
        rank, world_size = get_dist_info()
        output_path = self.output_file
        if not os.path.isabs(output_path):
            output_path = os.path.join(runner.work_dir, output_path)
        self.output_path = output_path
        if rank == 0:
            append_jsonl(
                self.output_path,
                {
                    "event": "run_start",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "world_size": world_size,
                    "model": model_statistics(runner.model),
                    "environment": runtime_environment(),
                },
            )
            runner.logger.info(
                "Training baseline metrics will be written to %s", self.output_path
            )

    def before_train_epoch(self, runner):
        self.local_samples = 0
        self.loss_sums = {}
        self.loss_weights = {}
        self.epoch_elapsed = None
        self.epoch_started_at = None
        self.train_peak_allocated = 0.0
        self.train_peak_reserved = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self, runner):
        if runner.inner_iter == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.epoch_started_at = time.perf_counter()

    def after_train_iter(self, runner):
        outputs = runner.outputs or {}
        samples = int(outputs.get("num_samples", 0))
        self.local_samples += samples
        for name, value in outputs.get("log_vars", {}).items():
            if isinstance(value, Number):
                self.loss_sums[name] = self.loss_sums.get(name, 0.0) + value * samples
                self.loss_weights[name] = self.loss_weights.get(name, 0) + samples

        if runner.inner_iter + 1 == len(runner.data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.epoch_elapsed = time.perf_counter() - self.epoch_started_at
            memory = cuda_memory()
            self.train_peak_allocated = memory.get("peak_allocated_mb", 0.0)
            self.train_peak_reserved = memory.get("peak_reserved_mb", 0.0)

    def after_train_epoch(self, runner):
        if self.epoch_elapsed is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self.epoch_started_at is None:
                self.epoch_started_at = time.perf_counter()
            self.epoch_elapsed = time.perf_counter() - self.epoch_started_at
            memory = cuda_memory()
            self.train_peak_allocated = memory.get("peak_allocated_mb", 0.0)
            self.train_peak_reserved = memory.get("peak_reserved_mb", 0.0)

        elapsed = _distributed_value(
            self.epoch_elapsed, torch.distributed.ReduceOp.MAX
        )
        global_samples = int(
            _distributed_value(self.local_samples, torch.distributed.ReduceOp.SUM)
        )
        peak_allocated = _distributed_value(
            self.train_peak_allocated,
            torch.distributed.ReduceOp.MAX,
        )
        peak_reserved = _distributed_value(
            self.train_peak_reserved,
            torch.distributed.ReduceOp.MAX,
        )

        rank, _ = get_dist_info()
        if rank != 0:
            return

        averages = {
            name: self.loss_sums[name] / max(self.loss_weights[name], 1)
            for name in sorted(self.loss_sums)
        }
        log_output = getattr(runner.log_buffer, "output", {})
        validation = {
            name: float(value)
            for name, value in log_output.items()
            if isinstance(value, Number)
            and (name.startswith("map/") or name.startswith("robotbev_"))
        }
        record = {
            "event": "train_epoch",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch": runner.epoch + 1,
            "global_samples": global_samples,
            "performance": {
                "train_epoch_seconds": elapsed,
                "train_ms_per_sample": elapsed * 1000 / max(global_samples, 1),
                "train_samples_per_second": global_samples / max(elapsed, 1e-12),
            },
            "loss": averages,
            "validation": validation,
            "memory": {
                **process_memory(),
                "cuda_peak_allocated_mb_max": peak_allocated,
                "cuda_peak_reserved_mb_max": peak_reserved,
            },
        }
        append_jsonl(self.output_path, record)
        runner.logger.info(
            "Baseline epoch %d: %.3f samples/s, %.1f MB peak CUDA, "
            "mIoU@0.50=%s",
            runner.epoch + 1,
            record["performance"]["train_samples_per_second"],
            peak_allocated,
            validation.get("robotbev_map_iou_50", "n/a"),
        )

    def after_run(self, runner):
        rank, _ = get_dist_info()
        if rank == 0:
            append_jsonl(
                self.output_path,
                {
                    "event": "run_end",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "epochs": runner.epoch,
                    "memory": process_memory(),
                },
            )
