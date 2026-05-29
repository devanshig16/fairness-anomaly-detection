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

        if self.strategy == "zscore":
            for grp, stats in self.group_stats_.items():
                mask = groups == grp
                debiased[mask] = (scores[mask] - stats["mean"]) / stats["std"]

        elif self.strategy == "quantile":
            # Rank-normalize within group to [0,1], then concatenate
            for grp, stats in self.group_stats_.items():
                mask = groups == grp
                s = scores[mask]
                ranks = s.argsort().argsort()
                debiased[mask] = ranks / (mask.sum() - 1 + 1e-9)

        return debiased

    def fit_transform(self, scores, groups):
        return self.fit(scores, groups).transform(scores, groups)

    def predict(self, debiased_scores: np.ndarray) -> np.ndarray:
        """Threshold debiased scores to produce binary labels."""
        threshold = np.percentile(debiased_scores, 100 * (1 - self.contamination))
        return (debiased_scores >= threshold).astype(int)


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
demographics = pd.read_csv(os.path.join(RESULTS_DIR, "demographics.csv"))
scores_df    = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_scores.csv"))
labels_df    = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_labels.csv"))

race_groups   = demographics["race"].values
gender_groups = demographics["gender"].values
detectors     = scores_df.columns.tolist()

# ── Apply FairnessWrapper (race axis) ─────────────────────────────────────────
print("\nApplying FairnessWrapper (zscore strategy) on race axis...")
wrapper = FairnessWrapper(contamination=CONTAMINATION, strategy="zscore")

debiased_scores = {}
debiased_labels = {}

for det in detectors:
    raw_scores = scores_df[det].values
    db_scores  = wrapper.fit_transform(raw_scores, race_groups)
    db_labels  = wrapper.predict(db_scores)
    # re-normalize debiased scores to [0,1] for comparability
    mn, mx = db_scores.min(), db_scores.max()
    debiased_scores[det] = (db_scores - mn) / (mx - mn + 1e-9)
    debiased_labels[det] = db_labels
    print(f"  {det}: {labels_df[det].sum()} → {db_labels.sum()} anomalies flagged")

debiased_scores_df = pd.DataFrame(debiased_scores)
debiased_labels_df = pd.DataFrame(debiased_labels)
debiased_scores_df.to_csv(os.path.join(RESULTS_DIR, "debiased_scores.csv"), index=False)
debiased_labels_df.to_csv(os.path.join(RESULTS_DIR, "debiased_labels.csv"), index=False)

# ── Compare fairness before vs after ─────────────────────────────────────────
print("\n── Fairness Comparison: Before vs After Debiasing ──")
race_values = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]

comparison = []
for det in detectors:
    for label_df, tag in [(labels_df, "before"), (debiased_labels_df, "after")]:
        rates = {}
        for grp in race_values:
            mask = demographics["race"] == grp
            if mask.sum() == 0:
                continue
            rates[grp] = label_df[det][mask].mean()
        spd = max(rates.values()) - min(rates.values())
        dir_r = min(rates.values()) / (max(rates.values()) + 1e-9)
        comparison.append({
            "detector": det,
            "stage":    tag,
            "SPD":      round(spd, 4),
            "DIR":      round(dir_r, 4),
            **{f"rate_{k}": round(v, 4) for k, v in rates.items()}
        })

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "fairness_comparison.csv"), index=False)

# Pretty print
print("\nSPD (Statistical Parity Difference) — lower is fairer:")
pivot = comparison_df.pivot(index="detector", columns="stage", values="SPD")
pivot["reduction_%"] = ((pivot["before"] - pivot["after"]) / (pivot["before"] + 1e-9) * 100).round(1)
print(pivot.to_string())

print("\nDIR (Disparate Impact Ratio) — closer to 1.0 is fairer:")
pivot2 = comparison_df.pivot(index="detector", columns="stage", values="DIR")
print(pivot2.to_string())

print("\n✓ Saved debiased_scores.csv")
print("✓ Saved debiased_labels.csv")
print("✓ Saved fairness_comparison.csv")
print("\nDebiasing complete.")
