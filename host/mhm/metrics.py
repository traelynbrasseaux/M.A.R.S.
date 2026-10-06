"""ROC metrics in plain numpy, with pAUC defined as in DCASE 2020 Task 2."""

import numpy as np


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates at every distinct threshold, starting from (0, 0)."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order].astype(bool)
    last_of_tie = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[last_of_tie]
    fps = last_of_tie + 1 - tps
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = roc_curve(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def partial_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """Area under the ROC for FPR in [0, max_fpr], divided by max_fpr (no McClish correction)."""
    fpr, tpr = roc_curve(labels, scores)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x0, x1, y0, y1 = fpr[stop - 1], fpr[stop], tpr[stop - 1], tpr[stop]
    tpr_at_max = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0)
    x = np.r_[fpr[:stop], max_fpr]
    y = np.r_[tpr[:stop], tpr_at_max]
    return float(np.trapezoid(y, x) / max_fpr)


def rates_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    """(detection rate, false alarm rate) when flagging scores above threshold."""
    flagged = scores > threshold
    labels = labels.astype(bool)
    return float(flagged[labels].mean()), float(flagged[~labels].mean())
