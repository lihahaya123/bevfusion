import json
import math
import os
import time
from numbers import Number

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info

from mmdet3d.utils.baseline_metrics import write_json


def _distributed_device():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return torch.device("cpu")
    if torch.distributed.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Stop epoch-based training when a validation metric stops improving."""

    def __init__(
        self,
        monitor,
        patience=5,
        min_delta=0.0,
        rule="greater",
        start_epoch=1,
        state_file="early_stopping_state.json",
    ):
        if rule not in {"greater", "less"}:
            raise ValueError("early stopping rule must be 'greater' or 'less'")
        if patience < 1:
            raise ValueError("early stopping patience must be at least 1")
        if min_delta < 0:
            raise ValueError("early stopping min_delta must be non-negative")
        if start_epoch < 1:
            raise ValueError("early stopping start_epoch must be at least 1")

        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.rule = rule
        self.start_epoch = int(start_epoch)
        self.state_file = state_file
        self.state_path = None
        self.best_score = -math.inf if rule == "greater" else math.inf
        self.best_epoch = 0
        self.bad_epochs = 0
        self._missing_metric_warned = False

    def before_run(self, runner):
        state_path = self.state_file
        if not os.path.isabs(state_path):
            state_path = os.path.join(runner.work_dir, state_path)
        self.state_path = state_path

        rank, _ = get_dist_info()
        if rank == 0:
            self._restore_state(runner)
        self._broadcast_state()
        if rank == 0:
            runner.logger.info(
                "Early stopping monitors %s (%s), patience=%d, min_delta=%g",
                self.monitor,
                self.rule,
                self.patience,
                self.min_delta,
            )

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch < self.start_epoch:
            return

        score = self._validation_score(runner)
        if score is None:
            rank, _ = get_dist_info()
            if rank == 0 and not self._missing_metric_warned:
                runner.logger.warning(
                    "Early stopping metric %s is unavailable; this epoch is ignored",
                    self.monitor,
                )
                self._missing_metric_warned = True
            return

        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.best_epoch = current_epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = self.bad_epochs >= self.patience
        rank, _ = get_dist_info()
        if rank == 0:
            if improved:
                runner.logger.info(
                    "Early stopping: %s improved to %.6f at epoch %d",
                    self.monitor,
                    score,
                    current_epoch,
                )
            else:
                runner.logger.info(
                    "Early stopping: %s=%.6f did not improve from %.6f "
                    "(%d/%d)",
                    self.monitor,
                    score,
                    self.best_score,
                    self.bad_epochs,
                    self.patience,
                )
            if should_stop:
                runner.logger.warning(
                    "Early stopping triggered at epoch %d; best %s=%.6f "
                    "at epoch %d",
                    current_epoch,
                    self.monitor,
                    self.best_score,
                    self.best_epoch,
                )
            self._write_state(
                current_epoch=current_epoch,
                score=score,
                stopped=should_stop,
            )

        if should_stop:
            runner._max_epochs = current_epoch

    def _validation_score(self, runner):
        rank, _ = get_dist_info()
        score = None
        if rank == 0:
            value = getattr(runner.log_buffer, "output", {}).get(self.monitor)
            if isinstance(value, Number):
                score = float(value)

        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return score

        device = _distributed_device()
        payload = torch.zeros(2, dtype=torch.float64, device=device)
        if rank == 0 and score is not None:
            payload[0] = 1
            payload[1] = score
        torch.distributed.broadcast(payload, src=0)
        return payload[1].item() if payload[0].item() else None

    def _is_improvement(self, score):
        if not math.isfinite(score):
            return False
        if self.rule == "greater":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def _restore_state(self, runner):
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, ValueError) as exc:
            runner.logger.warning(
                "Could not read early stopping state %s: %s",
                self.state_path,
                exc,
            )
            return

        if state.get("monitor") != self.monitor or state.get("rule") != self.rule:
            runner.logger.warning(
                "Ignoring incompatible early stopping state in %s",
                self.state_path,
            )
            return
        last_epoch = int(state.get("last_epoch", 0))
        if last_epoch != runner.epoch:
            runner.logger.warning(
                "Ignoring early stopping state from epoch %d because runner "
                "resumes at epoch %d",
                last_epoch,
                runner.epoch,
            )
            return

        best_score = state.get("best_score")
        if best_score is None:
            return
        self.best_score = float(best_score)
        self.best_epoch = int(state["best_epoch"])
        self.bad_epochs = int(state["bad_epochs"])
        runner.logger.info(
            "Restored early stopping state: best %.6f at epoch %d, bad epochs %d",
            self.best_score,
            self.best_epoch,
            self.bad_epochs,
        )

    def _broadcast_state(self):
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        rank, _ = get_dist_info()
        device = _distributed_device()
        payload = torch.zeros(3, dtype=torch.float64, device=device)
        if rank == 0:
            payload[0] = self.best_score
            payload[1] = self.best_epoch
            payload[2] = self.bad_epochs
        torch.distributed.broadcast(payload, src=0)
        self.best_score = payload[0].item()
        self.best_epoch = int(payload[1].item())
        self.bad_epochs = int(payload[2].item())

    def _write_state(self, current_epoch, score, stopped):
        write_json(
            self.state_path,
            {
                "schema_version": 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "monitor": self.monitor,
                "rule": self.rule,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "start_epoch": self.start_epoch,
                "last_epoch": current_epoch,
                "last_score": score if math.isfinite(score) else None,
                "best_score": (
                    self.best_score if math.isfinite(self.best_score) else None
                ),
                "best_epoch": self.best_epoch,
                "bad_epochs": self.bad_epochs,
                "stopped": stopped,
                "stop_epoch": current_epoch if stopped else None,
            },
        )
