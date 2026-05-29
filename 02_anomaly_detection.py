"""
02_anomaly_detection.py
-----------------------
Runs four unsupervised anomaly detectors on the preprocessed feature matrix.

Algorithms:
  - Isolation Forest (IF)
  - Local Outlier Factor (LOF)
  - One-Class SVM (OCSVM)
  - DBSCAN (via outlier score = distance to nearest cluster)

Output:
  - results/anomaly_scores.csv  — raw scores for each detector
  - results/anomaly_labels.csv  — binary flags (1 = anomaly) for each detector
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os, time

RESULTS_DIR = "results"
CONTAMINATION = 0.05   # assume 5% anomaly rate — standard for medical datasets
RANDOM_STATE  = 42

print("Loading preprocessed features...")
X = pd.read_csv(os.path.join(RESULTS_DIR, "processed_features.csv")).values
print(f"  Shape: {X.shape}")

scores = {}
labels = {}

# ── 1. Isolation Forest ───────────────────────────────────────────────────────
print("\n[1/4] Isolation Forest...")
t = time.time()
clf_if = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
clf_if.fit(X)
