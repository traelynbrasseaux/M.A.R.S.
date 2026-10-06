import numpy as np
import pytest

from mhm.metrics import partial_auc, rates_at_threshold, roc_auc


def test_auc_hand_example():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(labels, scores) == pytest.approx(0.75)


def test_auc_matches_pairwise_count():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    scores = rng.normal(size=300) + labels + np.round(rng.normal(size=300), 1)
    pos, neg = scores[labels == 1], scores[labels == 0]
    pairs = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    assert roc_auc(labels, scores) == pytest.approx(pairs.mean())


def test_partial_auc_extremes_and_ties():
    labels = np.array([0, 0, 0, 1, 1])
    assert partial_auc(labels, np.array([1, 2, 3, 4, 5.0])) == pytest.approx(1.0)
    assert partial_auc(labels, np.array([5, 4, 3, 2, 1.0])) == pytest.approx(0.0)
    # All tied gives the diagonal, whose area up to 0.1 is 0.005.
    assert partial_auc(labels, np.ones(5), max_fpr=0.1) == pytest.approx(0.05)


def test_rates_at_threshold():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])
    assert rates_at_threshold(labels, scores, 0.35) == (1.0, 0.25)
