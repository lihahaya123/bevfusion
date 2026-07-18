import torch

from mmdet3d.ops.spconv.modules import SparseSequential
from mmdet3d.ops.spconv.structure import SparseConvTensor


def test_sparse_batch_norm_uses_running_stats_for_single_feature():
    module = SparseSequential(torch.nn.BatchNorm1d(128))
    module.train()
    sparse = SparseConvTensor(
        features=torch.ones((1, 128)),
        indices=torch.zeros((1, 4), dtype=torch.int32),
        spatial_shape=[1, 1, 1],
        batch_size=1,
    )

    output = module(sparse)

    assert output.features.shape == (1, 128)
    assert torch.isfinite(output.features).all()
