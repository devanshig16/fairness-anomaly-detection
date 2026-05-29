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
# score_samples returns negative anomaly scores; negate so higher = more anomalous
scores["IF"] = -clf_if.score_samples(X)
labels["IF"] = (clf_if.predict(X) == -1).astype(int)
print(f"  Done in {time.time()-t:.1f}s  |  Anomalies flagged: {labels['IF'].sum()}")

# ── 2. Local Outlier Factor ───────────────────────────────────────────────────
print("\n[2/4] Local Outlier Factor...")
t = time.time()
clf_lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    n_jobs=-1
)
lof_labels = clf_lof.fit_predict(X)
# negative_outlier_factor_: more negative = more outlying; negate for consistency
scores["LOF"] = -clf_lof.negative_outlier_factor_
labels["LOF"] = (lof_labels == -1).astype(int)
print(f"  Done in {time.time()-t:.1f}s  |  Anomalies flagged: {labels['LOF'].sum()}")

# ── 3. One-Class SVM ─────────────────────────────────────────────────────────
print("\n[3/4] One-Class SVM (subsampled for speed)...")
t = time.time()
# OCSVM is O(n^2); subsample for training, score all
