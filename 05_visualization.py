"""
05_visualization.py
-------------------
Generates all figures for the paper.

Figure 1: Anomaly flagging rates by race group, per detector (grouped bar chart)
Figure 2: Anomaly score distributions by race group, per detector (violin plots)
Figure 3: Fairness metrics heatmap (SPD, DIR, JS divergence across detectors)
Figure 4: Before vs After debiasing — SPD reduction bar chart
Figure 5: Per-group anomaly rates before vs after debiasing (grouped)
Figure 6: Gender disparity — anomaly rates by detector
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

PALETTE_RACE   = ["#2C7BB6", "#D7191C", "#FDAE61", "#1A9641", "#7B3294"]
PALETTE_STAGE  = {"before": "#D7191C", "after": "#2C7BB6"}
RACE_LABELS    = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
RACE_SHORT     = ["Caucasian", "African\nAmerican", "Hispanic", "Asian", "Other"]
DETECTORS      = ["IF", "LOF", "OCSVM", "DBSCAN"]

# ── Load ──────────────────────────────────────────────────────────────────────
demographics   = pd.read_csv(os.path.join(RESULTS_DIR, "demographics.csv"))
scores_df      = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_scores.csv"))
labels_df      = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_labels.csv"))
race_rates     = pd.read_csv(os.path.join(RESULTS_DIR, "group_anomaly_rates_race.csv"))
gender_rates   = pd.read_csv(os.path.join(RESULTS_DIR, "group_anomaly_rates_gender.csv"))
