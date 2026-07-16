import torch

from mmdet3d.models.heads.segm.vanilla import masked_reduce_loss, sigmoid_focal_loss


def test_masked_focal_ignores_invalid_target_changes():
    logits = torch.tensor([[0.2, -0.7], [1.1, -1.3]], requires_grad=True)
    target_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_b = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss_a = masked_reduce_loss(
        sigmoid_focal_loss(logits, target_a, reduction="none"),
        mask,
    )
    loss_b = masked_reduce_loss(
        sigmoid_focal_loss(logits, target_b, reduction="none"),
        mask,
    )

    torch.testing.assert_close(loss_a, loss_b)


def test_masked_loss_zero_valid_pixels_keeps_graph():
    logits = torch.randn(2, 3, requires_grad=True)
    target = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)

    loss = masked_reduce_loss(
        sigmoid_focal_loss(logits, target, reduction="none"),
        mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad.abs().sum() == 0
