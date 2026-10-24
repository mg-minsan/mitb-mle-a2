"""Select a probability decision threshold from held-out validation data."""
import numpy as np
from sklearn.metrics import roc_curve


def select_decision_threshold(labels, probabilities):
    """Maximise TPR - FPR; ties favour the higher (more conservative) cutoff."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError('Labels and probabilities must be equally sized one-dimensional arrays')
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Threshold selection requires both binary classes 0 and 1')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be finite and between 0 and 1')
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    thresholds, objective = thresholds[valid], (tpr - fpr)[valid]
    return float(np.max(thresholds[objective == objective.max()]))
