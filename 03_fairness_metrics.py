"""
03_fairness_metrics.py
----------------------
Computes demographic disparity metrics for each anomaly detector.

Metrics (adapted to unsupervised setting — no ground-truth labels needed):
  - Statistical Parity Difference (SPD): max - min anomaly rate across groups
  - Disparate Impact Ratio (DIR):        min / max anomaly rate across groups
  - Mean Score Gap:                      max - min mean anomaly score across groups
  - KL Divergence:                       score distribution distance from majority group
  - Score Distribution Overlap:          how much score distributions overlap across groups

Outputs:
  - results/fairness_metrics_race.csv
  - results/fairness_metrics_gender.csv
  - results/group_anomaly_rates.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import os

RESULTS_DIR = "results"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading results...")
demographics = pd.read_csv(os.path.join(RESULTS_DIR, "demographics.csv"))
