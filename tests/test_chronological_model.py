import torch

from tscd_net.models.chronological import _pool_chronological_windows
from tscd_net.models.tscdnet import TSCDNet


def test_chronological_windows_ignore_right_padding() -> None:
    scores = torch.tensor([[[1., 2., 3., 4., 5., 90.]], [[6., 8., 90., 90., 90., 90.]]])
    gaps = torch.zeros(2, 6)
    lengths = torch.tensor([5, 2])
    pooled, pooled_gaps, counts = _pool_chronological_windows(scores, gaps, lengths, 3)
    torch.testing.assert_close(pooled, torch.tensor([[[1., 2.5, 4.5]], [[6., 8., 0.]]]))
    torch.testing.assert_close(pooled_gaps, torch.zeros(2, 3))
    torch.testing.assert_close(counts, torch.tensor([3, 2]))


def test_chronological_model_distinguishes_reversed_valid_posts() -> None:
    torch.manual_seed(42)
    model = TSCDNet(
        torch.randn(5, 8), model_dim=16, layers=1, dropout=0.0,
        chronological_windows=4, temporal_encoding=False,
    ).eval()
    scores = torch.rand(1, 5, 13)
    gaps = torch.zeros(1, 13)
    lengths = torch.tensor([12])
    reversed_scores = scores.clone()
    reversed_scores[:, :, :12] = scores[:, :, :12].flip(-1)
    updated_padding = scores.clone()
    updated_padding[:, :, 12:] = 100
    with torch.inference_mode():
        original = model(scores, gaps, lengths)
        reversed_output = model(reversed_scores, gaps, lengths)
        padded_output = model(updated_padding, gaps, lengths)
    assert (original - reversed_output).abs().item() > 1e-5
    torch.testing.assert_close(original, padded_output)
