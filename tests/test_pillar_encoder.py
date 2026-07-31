import torch

from mmdet3d.models.backbones.pillar_encoder import (
    PillarBEVEncoder,
    PointPillarsEncoder,
)


def test_pillar_bev_encoder_downsamples_by_four():
    encoder = PillarBEVEncoder(
        in_channels=8,
        channels=[8, 16, 16],
        strides=[2, 2, 1],
    )
    encoder.eval()

    with torch.no_grad():
        output = encoder(torch.randn(2, 8, 32, 32))

    assert output.shape == (2, 16, 8, 8)


def test_point_pillars_encoder_applies_bev_encoder_after_scatter():
    encoder = PointPillarsEncoder(
        pts_voxel_encoder=dict(
            type="PillarFeatureNet",
            in_channels=5,
            feat_channels=[8],
            with_distance=False,
            voxel_size=[0.01, 0.01, 2.5],
            point_cloud_range=[0.0, -1.52, -0.5, 3.04, 1.52, 2.0],
        ),
        pts_middle_encoder=dict(
            type="PointPillarsScatter",
            in_channels=8,
            output_shape=[32, 32],
        ),
        pts_bev_encoder=dict(
            type="PillarBEVEncoder",
            in_channels=8,
            channels=[8, 16, 16],
            strides=[2, 2, 1],
        ),
    )
    encoder.train()

    features = torch.zeros(4, 4, 5)
    sizes = torch.tensor([2, 1, 3, 4], dtype=torch.int32)
    for pillar_index, point_count in enumerate(sizes.tolist()):
        features[pillar_index, :point_count, :3] = torch.randn(point_count, 3)
    coords = torch.tensor(
        [
            [0, 1, 2, 0],
            [0, 4, 5, 0],
            [1, 8, 9, 0],
            [1, 12, 13, 0],
        ],
        dtype=torch.int32,
    )

    output = encoder(features, coords, batch_size=2, sizes=sizes)
    output.mean().backward()

    assert output.shape == (2, 16, 8, 8)
    assert encoder.pts_voxel_encoder.pfn_layers[0].linear.weight.grad is not None
    assert encoder.pts_bev_encoder.blocks[0].depthwise.weight.grad is not None


def test_point_pillars_encoder_keeps_scatter_output_without_bev_encoder():
    encoder = PointPillarsEncoder(
        pts_voxel_encoder=dict(
            type="PillarFeatureNet",
            in_channels=5,
            feat_channels=[8],
            voxel_size=[0.01, 0.01, 2.5],
            point_cloud_range=[0.0, -1.52, -0.5, 3.04, 1.52, 2.0],
        ),
        pts_middle_encoder=dict(
            type="PointPillarsScatter",
            in_channels=8,
            output_shape=[16, 16],
        ),
    )
    encoder.eval()

    features = torch.randn(2, 2, 5)
    sizes = torch.tensor([2, 2], dtype=torch.int32)
    coords = torch.tensor(
        [[0, 1, 2, 0], [0, 4, 5, 0]],
        dtype=torch.int32,
    )

    with torch.no_grad():
        output = encoder(features, coords, batch_size=1, sizes=sizes)

    assert output.shape == (1, 8, 16, 16)
