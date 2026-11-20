import numpy as np

from tscd_net.splits import stratified_70_10_20


def test_stratified_split_is_disjoint_and_preserves_both_classes() -> None:
    labels = np.asarray([0] * 100 + [1] * 40)
    split = stratified_70_10_20(labels, seed=7)
    all_indices = np.concatenate([split.train, split.validation, split.test])
    assert len(all_indices) == len(labels)
    assert len(np.unique(all_indices)) == len(labels)
    for subset in (split.train, split.validation, split.test):
        assert set(labels[subset]) == {0, 1}
    assert abs(len(split.train) / len(labels) - 0.7) < 0.01
    assert abs(len(split.validation) / len(labels) - 0.1) < 0.02
