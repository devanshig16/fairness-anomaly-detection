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
np.random.seed(RANDOM_STATE)
n_train = min(10000, len(X))
idx_train = np.random.choice(len(X), n_train, replace=False)
clf_ocsvm = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
clf_ocsvm.fit(X[idx_train])
# decision_function: negative = anomaly; negate for consistency
raw_ocsvm = -clf_ocsvm.decision_function(X)
scores["OCSVM"] = raw_ocsvm
threshold_ocsvm = np.percentile(raw_ocsvm, 100 * (1 - CONTAMINATION))
labels["OCSVM"] = (raw_ocsvm >= threshold_ocsvm).astype(int)
print(f"  Done in {time.time()-t:.1f}s  |  Anomalies flagged: {labels['OCSVM'].sum()}")

# ── 4. DBSCAN ────────────────────────────────────────────────────────────────
print("\n[4/4] DBSCAN...")
t = time.time()
# Use k-NN distance as the anomaly score (distance to 5th nearest neighbor)
# Points in dense regions get low scores; isolated points get high scores
knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
knn.fit(X)
knn_distances, _ = knn.kneighbors(X)
scores["DBSCAN"] = knn_distances[:, -1]   # distance to 5th neighbor

# DBSCAN labels: -1 = noise (anomaly)
