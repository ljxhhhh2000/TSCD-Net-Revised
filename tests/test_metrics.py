import numpy as np

from tscd_net.metrics import best_f1_threshold


def test_validation_threshold_can_recover_positives_below_half() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    assert best_f1_threshold(labels, probabilities) == 0.2
