import numpy as np
import pytest

from mhm.models import AnomalyModel
from mhm.models.hdc import HdcScorer, item_memory, lfsr_bits
from mhm.models.mahalanobis import Mahalanobis

N_FEATURES = 16


def synthetic(n: int, rng: np.random.Generator, mixing: np.ndarray) -> np.ndarray:
    return rng.normal(size=(n, N_FEATURES)) @ mixing + 5.0


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    mixing = np.eye(N_FEATURES) + 0.3 * rng.normal(size=(N_FEATURES, N_FEATURES))
    train = synthetic(3000, rng, mixing)
    normal = synthetic(500, rng, mixing)
    anomalous = synthetic(500, rng, mixing)
    anomalous[:, [2, 7, 11]] += 4 * train[:, [2, 7, 11]].std(axis=0)
    return train, normal, anomalous


@pytest.mark.parametrize(
    "model",
    [Mahalanobis(), HdcScorer(mode="binary"), HdcScorer(mode="bipolar")],
    ids=["mahalanobis", "hdc-binary", "hdc-bipolar"],
)
def test_normal_low_anomaly_high(model: AnomalyModel, data):
    train, normal, anomalous = data
    model.fit(train)
    normal_scores, anomaly_scores = model.score(normal), model.score(anomalous)
    threshold = np.quantile(normal_scores, 0.95)
    assert np.median(anomaly_scores) > threshold
    assert np.mean(anomaly_scores > threshold) > 0.9


def test_lfsr_reference_stream():
    bits = lfsr_bits(1, 32)
    assert int("".join(map(str, bits[::-1])), 2) == 0x8A2DB6DB
    with pytest.raises(ValueError):
        lfsr_bits(0, 8)


def test_item_memory_geometry():
    dim, n_levels = 4096, 16
    mem = item_memory(0xBEEF, dim, n_levels, n_features=4)
    from_base = np.count_nonzero(mem.levels != mem.levels[0], axis=1) / dim
    assert np.all(np.diff(from_base) > 0)
    assert abs(from_base[-1] - 0.5) < 0.03
    assert abs(from_base[n_levels // 2] - 0.5 * 8 / 15) < 0.03
    ids_distance = np.count_nonzero(mem.ids[0] != mem.ids[1]) / dim
    assert abs(ids_distance - 0.5) < 0.03


def test_hdc_is_reproducible_from_seed(data):
    train, normal, _ = data
    a = HdcScorer(seed=7).fit(train).score(normal)
    b = HdcScorer(seed=7).fit(train).score(normal)
    c = HdcScorer(seed=8).fit(train).score(normal)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
