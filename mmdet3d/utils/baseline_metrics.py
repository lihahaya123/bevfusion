import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch


MEBIBYTE = 1024 * 1024


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def model_statistics(model):
    model = unwrap_model(model)
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_bytes = sum(item.numel() * item.element_size() for item in parameters)
    buffer_bytes = sum(item.numel() * item.element_size() for item in buffers)
    return {
        "parameters": sum(item.numel() for item in parameters),
        "trainable_parameters": sum(
            item.numel() for item in parameters if item.requires_grad
        ),
        "parameter_size_mb": parameter_bytes / MEBIBYTE,
        "buffer_size_mb": buffer_bytes / MEBIBYTE,
        "model_state_size_mb": (parameter_bytes + buffer_bytes) / MEBIBYTE,
    }


def runtime_environment():
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "index": device,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mb": properties.total_memory / MEBIBYTE,
        }
    return result


def process_memory():
    values = {}
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM"}:
                    values[key] = float(value.strip().split()[0]) / 1024
    except OSError:
        pass
    return {
        "rss_mb": values.get("VmRSS"),
        "peak_rss_mb": values.get("VmHWM"),
    }


def cuda_memory():
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    return {
        "device_index": device,
        "allocated_mb": torch.cuda.memory_allocated(device) / MEBIBYTE,
        "reserved_mb": torch.cuda.memory_reserved(device) / MEBIBYTE,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / MEBIBYTE,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / MEBIBYTE,
    }


def summarize_samples(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(ratio):
        if len(ordered) == 1:
            return ordered[0]
        position = ratio * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")
        handle.flush()


class InferenceLatencyRecorder:
    """Record top-level model latency without changing model outputs."""

    def __init__(self, model, warmup=5):
        self.model = model
        self.warmup = max(int(warmup), 0)
        self._starts = []
        self._pairs = []
        self._cpu_start = None
        self._cpu_values = []
        self._handles = []

    def start(self):
        self._handles = [
            self.model.register_forward_pre_hook(self._before_forward),
            self.model.register_forward_hook(self._after_forward),
        ]
        return self

    def stop(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _before_forward(self, _module, _inputs):
        if torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._starts.append(event)
        else:
            self._cpu_start = time.perf_counter()

    def _after_forward(self, _module, _inputs, _output):
        if torch.cuda.is_available():
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pairs.append((self._starts.pop(), end))
        elif self._cpu_start is not None:
            self._cpu_values.append((time.perf_counter() - self._cpu_start) * 1000)
            self._cpu_start = None

    def summary(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            values = [start.elapsed_time(end) for start, end in self._pairs]
        else:
            values = self._cpu_values
        measured = values[self.warmup :]
        result = summarize_samples(measured)
        result["unit"] = "ms_per_batch"
        result["warmup_batches"] = min(self.warmup, len(values))
        result["total_batches"] = len(values)
        return result
