import torch

from mmdet3d.models.fusion_models.bevfusion import BEVFusion


class FakeVoxelizer(torch.nn.Module):
    max_num_points = 10

    def __call__(self, points):
        if points.shape[0] == 0:
            raise RuntimeError("empty points should be handled before voxelizer")
        return (
            points.new_zeros((1, self.max_num_points, points.shape[1])),
            torch.zeros((1, 3), device=points.device, dtype=torch.int),
            torch.ones((1,), device=points.device, dtype=torch.int),
        )


def test_voxelize_handles_empty_point_tensor():
    model = BEVFusion.__new__(BEVFusion)
    torch.nn.Module.__init__(model)
    model.encoders = torch.nn.ModuleDict(
        {"lidar": torch.nn.ModuleDict({"voxelize": FakeVoxelizer()})}
    )
    model.voxelize_reduce = True

    feats, coords, sizes = model.voxelize(
        [torch.empty((0, 5)), torch.ones((3, 5))],
        "lidar",
    )

    assert feats.shape == (4, 5)
    assert coords.shape == (4, 4)
    assert sizes.shape == (4,)
    assert coords[:, 0].tolist() == [0, 0, 1, 1]
