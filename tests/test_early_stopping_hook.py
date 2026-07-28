import json
import logging
from types import SimpleNamespace

import pytest

from mmdet3d.runner import EarlyStoppingHook


def _runner(tmp_path):
    return SimpleNamespace(
        work_dir=str(tmp_path),
        logger=logging.getLogger("early-stopping-test"),
        epoch=0,
        _max_epochs=20,
        log_buffer=SimpleNamespace(output={}),
    )


def test_early_stopping_stops_after_patience(tmp_path):
    runner = _runner(tmp_path)
    hook = EarlyStoppingHook(
        monitor="robotbev_map_iou_max",
        patience=2,
        min_delta=0.001,
    )
    hook.before_run(runner)

    for epoch, score in enumerate([0.10, 0.12, 0.119, 0.118]):
        runner.epoch = epoch
        runner.log_buffer.output = {"robotbev_map_iou_max": score}
        hook.after_train_epoch(runner)

    assert runner._max_epochs == 4
    assert hook.best_score == pytest.approx(0.12)
    assert hook.best_epoch == 2
    state = json.loads((tmp_path / "early_stopping_state.json").read_text())
    assert state["bad_epochs"] == 2
    assert state["stopped"] is True
    assert state["stop_epoch"] == 4


def test_early_stopping_ignores_epochs_without_validation(tmp_path):
    runner = _runner(tmp_path)
    hook = EarlyStoppingHook(
        monitor="robotbev_map_iou_max",
        patience=1,
    )
    hook.before_run(runner)
    hook.after_train_epoch(runner)

    assert runner._max_epochs == 20
    assert hook.bad_epochs == 0


def test_early_stopping_restores_matching_resume_state(tmp_path):
    runner = _runner(tmp_path)
    first_hook = EarlyStoppingHook(
        monitor="robotbev_map_iou_max",
        patience=3,
    )
    first_hook.before_run(runner)
    runner.log_buffer.output = {"robotbev_map_iou_max": 0.2}
    first_hook.after_train_epoch(runner)

    resumed_runner = _runner(tmp_path)
    resumed_runner.epoch = 1
    resumed_hook = EarlyStoppingHook(
        monitor="robotbev_map_iou_max",
        patience=3,
    )
    resumed_hook.before_run(resumed_runner)

    assert resumed_hook.best_score == pytest.approx(0.2)
    assert resumed_hook.best_epoch == 1
    assert resumed_hook.bad_epochs == 0
