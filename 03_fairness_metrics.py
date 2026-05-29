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
scores_df    = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_scores.csv"))
labels_df    = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_labels.csv"))

detectors = scores_df.columns.tolist()
print(f"  Detectors: {detectors}")
print(f"  N = {len(demographics)}")

# ── Helper: KL divergence between two score distributions ─────────────────────
def kl_divergence(scores_a, scores_b, bins=50):
    """Jensen-Shannon divergence (symmetric, bounded [0,1]) between two score arrays."""
    hist_range = (0, 1)
    p, _ = np.histogram(scores_a, bins=bins, range=hist_range, density=True)
    q, _ = np.histogram(scores_b, bins=bins, range=hist_range, density=True)
    p = p + 1e-10
    q = q + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def compute_fairness(group_col, group_values, detectors, demographics, scores_df, labels_df):
    """Compute all fairness metrics for a given demographic axis."""
    results = []
    rate_records = []

    for detector in detectors:
        scores = scores_df[detector].values
        labels = labels_df[detector].values

        group_rates  = {}
        group_means  = {}
        group_scores = {}

        for grp in group_values:
            mask = (demographics[group_col] == grp).values
            if mask.sum() == 0:
                continue
            rate = labels[mask].mean()
            mean_score = scores[mask].mean()
            group_rates[grp]  = rate
            group_means[grp]  = mean_score
            group_scores[grp] = scores[mask]

            rate_records.append({
                "detector":    detector,
                group_col:     grp,
                "n":           mask.sum(),
                "anomaly_rate": rate,
                "mean_score":  mean_score
            })

        if len(group_rates) < 2:
            continue

        rates = list(group_rates.values())
        means = list(group_means.values())

        spd = max(rates) - min(rates)
        dir_ratio = min(rates) / (max(rates) + 1e-9)
        score_gap = max(means) - min(means)

        # KL vs majority group (Caucasian for race, Female for gender)
        majority = group_values[0]
        kl_divs = []
        for grp in group_values[1:]:
            if grp in group_scores and majority in group_scores:
                kl_divs.append(kl_divergence(group_scores[majority], group_scores[grp]))
        mean_kl = np.mean(kl_divs) if kl_divs else 0.0

        results.append({
            "detector":             detector,
            "SPD":                  round(spd, 4),
            "DIR":                  round(dir_ratio, 4),
            "mean_score_gap":       round(score_gap, 4),
            "mean_JS_divergence":   round(mean_kl, 4),
            "max_anomaly_rate":     round(max(rates), 4),
            "min_anomaly_rate":     round(min(rates), 4),
            "highest_rate_group":   max(group_rates, key=group_rates.get),
            "lowest_rate_group":    min(group_rates, key=group_rates.get),
        })

    return pd.DataFrame(results), pd.DataFrame(rate_records)


# ── Race analysis ─────────────────────────────────────────────────────────────
print("\n── Race Fairness Analysis ──")
race_groups = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
race_metrics, race_rates = compute_fairness(
    "race", race_groups, detectors, demographics, scores_df, labels_df
)
print(race_metrics.to_string(index=False))
race_metrics.to_csv(os.path.join(RESULTS_DIR, "fairness_metrics_race.csv"), index=False)
race_rates.to_csv(os.path.join(RESULTS_DIR, "group_anomaly_rates_race.csv"), index=False)

# ── Gender analysis ───────────────────────────────────────────────────────────
print("\n── Gender Fairness Analysis ──")
gender_groups = ["Female", "Male"]
gender_metrics, gender_rates = compute_fairness(
    "gender", gender_groups, detectors, demographics, scores_df, labels_df
)
print(gender_metrics.to_string(index=False))
gender_metrics.to_csv(os.path.join(RESULTS_DIR, "fairness_metrics_gender.csv"), index=False)
gender_rates.to_csv(os.path.join(RESULTS_DIR, "group_anomaly_rates_gender.csv"), index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Key Findings ──")
print("Race — highest SPD (worst fairness):")
print(race_metrics.sort_values("SPD", ascending=False)[["detector","SPD","DIR","highest_rate_group","lowest_rate_group"]].head())
print("\nGender — SPD per detector:")
print(gender_metrics[["detector","SPD","DIR","highest_rate_group"]].to_string(index=False))

print("\n✓ Saved fairness_metrics_race.csv")
print("✓ Saved fairness_metrics_gender.csv")
print("✓ Saved group_anomaly_rates_race.csv / _gender.csv")
print("\nFairness metrics complete.")
