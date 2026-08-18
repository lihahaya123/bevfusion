import numpy as np
import torch

from mmdet3d.core.points import LiDARPoints
from tools.train_lidar_degradation import RandomLidarDegradation


def make_grid_points():
    x = torch.linspace(0.0, 2.0, 41)
    y = torch.linspace(-1.0, 1.0, 41)
    grid_x, grid_y = torch.meshgrid(x, y)
    tensor = torch.zeros((grid_x.numel(), 5), dtype=torch.float32)
    tensor[:, 0] = grid_x.reshape(-1)
    tensor[:, 1] = grid_y.reshape(-1)
    return LiDARPoints(tensor, points_dim=5)


def test_random_local_box_removes_points_and_preserves_gt():
    np.random.seed(0)
    points = make_grid_points()
    gt = np.ones((6, 10, 10), dtype=np.uint8)
    transform = RandomLidarDegradation(
        point_cloud_range=[0.0, -1.0, -1.0, 2.0, 1.0, 1.0],
        mode_probabilities=[0.0, 1.0, 0.0, 0.0],
        local_size_x=[0.5, 0.5],
        local_size_y=[0.5, 0.5],
        two_box_prob=0.0,
        min_points=1,
    )

    result = transform({"points": points, "gt_masks_bev": gt.copy()})

    info = result["lidar_degradation"]
    box = info["boxes"][0]
    kept = result["points"].tensor
    inside = (
        (kept[:, 0] >= box["x_min"])
        & (kept[:, 0] <= box["x_max"])
        & (kept[:, 1] >= box["y_min"])
        & (kept[:, 1] <= box["y_max"])
    )
    assert info["mode"] == "local"
    assert info["retained_points"] < info["original_points"]
    assert not torch.any(inside)
    assert np.array_equal(result["gt_masks_bev"], gt)


def test_global_drop_keeps_requested_fraction_without_empty_lidar():
    np.random.seed(0)
    torch.manual_seed(0)
    tensor = torch.zeros((1000, 5), dtype=torch.float32)
    tensor[:, 0] = torch.linspace(0.0, 2.0, 1000)
    points = LiDARPoints(tensor, points_dim=5)
    transform = RandomLidarDegradation(
        point_cloud_range=[0.0, -1.0, -1.0, 2.0, 1.0, 1.0],
        mode_probabilities=[0.0, 0.0, 1.0, 0.0],
        global_drop_ratio=[0.3, 0.3],
        min_points=128,
    )

    result = transform({"points": points})

    info = result["lidar_degradation"]
    assert info["mode"] == "global"
    assert info["original_points"] == 1000
    assert info["retained_points"] == 700
    assert len(result["points"]) == 700
