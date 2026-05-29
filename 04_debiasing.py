"""
04_debiasing.py
---------------
The FairnessWrapper: a lightweight, algorithm-agnostic post-processing
debiasing method for unsupervised anomaly detectors.

Method:
  Given raw anomaly scores from any detector:
  1. Compute per-group score distributions
  2. Normalize scores within each demographic group (z-score normalization)
  3. Re-threshold using a global percentile on the normalized scores
  -> Groups now have equalized anomaly flagging rates

This requires ONLY the demographic labels at inference time (not training time),
making it compatible with any existing anomaly detection pipeline.

Outputs:
  - results/debiased_scores.csv
  - results/debiased_labels.csv
  - results/fairness_comparison.csv   (before vs after debiasing)
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR  = "results"
CONTAMINATION = 0.05


class FairnessWrapper:
    """
    Algorithm-agnostic post-processing wrapper for fairness in anomaly detection.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the dataset.
    strategy : str
        'zscore'    — normalize scores within each group, then threshold globally
        'quantile'  — equalize flagging rates by applying per-group quantile thresholds
    """

    def __init__(self, contamination=0.05, strategy="zscore"):
        self.contamination = contamination
        self.strategy = strategy
        self.group_stats_ = {}

    def fit(self, scores: np.ndarray, groups: np.ndarray):
        """Learn per-group score statistics."""
        self.group_stats_ = {}
        for grp in np.unique(groups):
            mask = groups == grp
            self.group_stats_[grp] = {
                "mean": scores[mask].mean(),
                "std":  scores[mask].std() + 1e-9,
                "q_threshold": np.percentile(scores[mask], 100 * (1 - self.contamination))
            }
        return self

    def transform(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Return debiased anomaly scores."""
        debiased = np.zeros_like(scores, dtype=float)

