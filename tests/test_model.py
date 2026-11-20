import torch

from tscd_net.models.time2vec import Time2Vec
from tscd_net.models.tscdnet import TSCDNet


def test_time2vec_shape_is_finite() -> None:
    module = Time2Vec(16)
    gaps = torch.zeros(2, 12)
    output = module(gaps)
    assert output.shape == (2, 12, 16)
    assert torch.isfinite(output).all()


def test_tscdnet_forward_and_masking() -> None:
    torch.manual_seed(4)
    model = TSCDNet(
        symptom_embeddings=torch.randn(15, 8),
        model_dim=32,
        time_dim=16,
        heads=4,
        layers=1,
        dropout=0.1,
        ffn_multiplier=2,
    )
    scores = torch.rand(3, 15, 20)
    gaps = torch.zeros(3, 20)
    lengths = torch.tensor([5, 12, 20])
    logits = model(scores, gaps, lengths)
    assert logits.shape == (3, 1)
    assert torch.isfinite(logits).all()

    masked_model = TSCDNet(
        symptom_embeddings=torch.randn(15, 8),
        model_dim=16,
        time_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        symptom_attention=False,
        temporal_attention=False,
        temporal_encoding=False,
    )
    masked_model.eval()
    original = scores.clone()
    changed = scores.clone()
    changed[0, :, 5:] = 100.0
    changed[1, :, 12:] = 100.0
    first = masked_model(original, gaps, lengths)
    second = masked_model(changed, gaps, lengths)
    assert torch.allclose(first, second)
