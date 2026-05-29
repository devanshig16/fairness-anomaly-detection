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
