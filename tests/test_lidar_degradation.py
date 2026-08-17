from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tools.test_lidar_degradation import (
    LidarDegrader,
    LidarDegradationTester,
    calculate_bev_metrics,
    draw_missing_region_box,
    replace_points_in_data,
)


class DummyDataContainer:
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


def test_region_degradation_sparsely_keeps_points_and_builds_bev_mask():
    inside = torch.zeros((10, 5))
    inside[:, 0] = 1.5
    inside[:, 1] = 0.0
    outside = torch.tensor(
        [
            [0.5, 0.0, 0.0, 1.0, 0.0],
            [1.5, 1.0, 0.0, 1.0, 0.0],
        ]
    )
    points = torch.cat([inside, outside], dim=0)
    degrader = LidarDegrader("region_mask", region_keep_ratio=0.3)
    torch.manual_seed(0)

    degraded = degrader.degrade(points)
    missing_mask = degrader.missing_bev_mask(150, 150)

    assert degraded.shape == (5, 5)
    assert degrader.count_region_points(degraded) == 3
    assert torch.any(torch.all(degraded == outside[0], dim=1))
    assert torch.any(torch.all(degraded == outside[1], dim=1))
    assert missing_mask.shape == (150, 150)
    assert missing_mask.sum() == 2500


def test_full_removal_returns_a_genuinely_empty_cloud():
    points = torch.ones((8, 5))

    empty = LidarDegrader("full_lidar_remove").degrade(points)

    assert empty.shape == (0, 5)
    assert empty.dtype == points.dtype


def test_replace_points_preserves_datacontainer_batch_nesting():
    original = torch.ones((8, 5))
    empty = original[:0].clone()
    data = {"points": DummyDataContainer([[original]])}

    replaced = replace_points_in_data(data, empty)

    assert isinstance(replaced["points"].data, list)
    assert isinstance(replaced["points"].data[0], list)
    assert replaced["points"].data[0][0].shape == (0, 5)
    assert data["points"].data[0][0].shape == (8, 5)


def test_missing_region_box_is_drawn_on_bev_png():
    with TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "bev.png"
        cv2.imwrite(str(image_path), np.full((20, 20, 3), 255, dtype=np.uint8))
        missing_mask = np.zeros((20, 20), dtype=bool)
        missing_mask[5:15, 6:14] = True

        draw_missing_region_box(image_path, missing_mask)

        rendered = cv2.imread(str(image_path))
        assert rendered[5, 6].tolist() == [0, 0, 255]


def test_visualization_saves_only_four_png_outputs():
    with TemporaryDirectory() as temp_dir:
        tester = object.__new__(LidarDegradationTester)
        tester.show_dir = Path(temp_dir)
        tester.map_classes = [
            "floor",
            "carpet",
            "wall",
            "furniture",
            "door",
            "clutter",
        ]
        tester.args = SimpleNamespace(map_threshold=0.5)
        scores = np.zeros((6, 8, 8), dtype=np.float32)
        gt = np.zeros((6, 8, 8), dtype=bool)
        output = {"masks_bev": scores}
        result = {
            "original": {"masks_bev": scores, "gt_masks_bev": gt},
            "degraded": output,
            "camera_only": output,
            "camera_ablated_degraded": output,
            "missing_bev_mask": np.ones((8, 8), dtype=bool),
        }

        tester.save_bev_visualizations(0, result)

        filenames = {
            path.name for path in (Path(temp_dir) / "sample_000000").iterdir()
        }
        assert filenames == {
            "gt_masks_bev.png",
            "original_masks_bev.png",
            "sparse_masks_bev.png",
            "sparse_camera_ablated_masks_bev.png",
        }


def test_bev_metrics_use_model_probabilities_without_a_second_sigmoid():
    scores = np.zeros((1, 2, 2), dtype=np.float32)
    target = np.ones((1, 2, 2), dtype=np.uint8)

    metrics = calculate_bev_metrics(
        scores,
        target,
        ["floor"],
        threshold=0.5,
    )

    assert metrics["mean_iou"] == 0.0
    assert metrics["counts"]["tp"] == [0]
    assert metrics["counts"]["fn"] == [4]
