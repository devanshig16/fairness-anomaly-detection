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
race_metrics   = pd.read_csv(os.path.join(RESULTS_DIR, "fairness_metrics_race.csv"))
gender_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "fairness_metrics_gender.csv"))
comparison_df  = pd.read_csv(os.path.join(RESULTS_DIR, "fairness_comparison.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Anomaly flagging rates by race per detector
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
fig.suptitle("Figure 1: Anomaly Flagging Rates by Race Group", fontsize=13, fontweight="bold", y=1.02)

for ax, det in zip(axes, DETECTORS):
    sub = race_rates[race_rates["detector"] == det]
    sub = sub.set_index("race").reindex(RACE_LABELS).reset_index()
    bars = ax.bar(range(len(RACE_LABELS)), sub["anomaly_rate"], color=PALETTE_RACE, edgecolor="white", linewidth=0.5)
    ax.set_title(det, fontweight="bold")
    ax.set_xticks(range(len(RACE_LABELS)))
    ax.set_xticklabels(RACE_SHORT, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Anomaly Rate" if det == "IF" else "")
    ax.set_ylim(0, min(1.0, sub["anomaly_rate"].max() * 1.4 + 0.01))
    # Annotate bars
    for bar, rate in zip(bars, sub["anomaly_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_race_flagging_rates.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "fig1_race_flagging_rates.png"), bbox_inches="tight")
plt.close()
print("✓ Figure 1 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Score distributions by race (violin plots)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
fig.suptitle("Figure 2: Anomaly Score Distributions by Race Group", fontsize=13, fontweight="bold", y=1.02)

for ax, det in zip(axes, DETECTORS):
    plot_data = []
    for grp in RACE_LABELS:
        mask = demographics["race"] == grp
        s = scores_df[det][mask.values].values
        for val in s:
            plot_data.append({"Race": grp.replace("AfricanAmerican","African\nAmerican"), "Score": val})
    plot_df = pd.DataFrame(plot_data)
    sns.violinplot(data=plot_df, x="Race", y="Score", ax=ax,
                   palette=PALETTE_RACE, inner="quartile", linewidth=0.8, cut=0)
    ax.set_title(det, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Anomaly Score (normalized)" if det == "IF" else "")
    ax.tick_params(axis="x", labelsize=8, rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_score_distributions.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "fig2_score_distributions.png"), bbox_inches="tight")
plt.close()
print("✓ Figure 2 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Fairness metrics heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
fig.suptitle("Figure 3: Fairness Metrics Across Detectors (Race Axis)", fontsize=13, fontweight="bold")

metrics_to_plot = [
    ("SPD",                "Statistical Parity Difference\n(lower = fairer)"),
    ("DIR",                "Disparate Impact Ratio\n(closer to 1 = fairer)"),
    ("mean_JS_divergence", "Mean JS Divergence\n(lower = fairer)"),
]

for ax, (metric, title) in zip(axes, metrics_to_plot):
    vals = race_metrics.set_index("detector")[metric].reindex(DETECTORS).values.reshape(-1, 1)
    im = ax.imshow(vals, cmap="RdYlGn_r" if metric != "DIR" else "RdYlGn",
                   aspect="auto", vmin=0, vmax=vals.max()*1.1)
    ax.set_xticks([])
    ax.set_yticks(range(len(DETECTORS)))
    ax.set_yticklabels(DETECTORS, fontsize=11)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.05)
    for i, v in enumerate(vals.flatten()):
        ax.text(0, i, f"{v:.3f}", ha="center", va="center", fontsize=11,
                color="white" if v > vals.max()*0.6 else "black", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_fairness_heatmap.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "fig3_fairness_heatmap.png"), bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: SPD before vs after debiasing
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Figure 4: SPD Before vs After FairnessWrapper Debiasing", fontsize=13, fontweight="bold")

x = np.arange(len(DETECTORS))
w = 0.35
for i, stage in enumerate(["before", "after"]):
    sub = comparison_df[comparison_df["stage"] == stage].set_index("detector").reindex(DETECTORS)
    bars = ax.bar(x + i*w, sub["SPD"], w, label=stage.capitalize(),
                  color=PALETTE_STAGE[stage], edgecolor="white")
    for bar, v in zip(bars, sub["SPD"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + w/2)
ax.set_xticklabels(DETECTORS)
ax.set_ylabel("Statistical Parity Difference (SPD)")
ax.legend()
ax.set_ylim(0, comparison_df["SPD"].max() * 1.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_debiasing_spd.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "fig4_debiasing_spd.png"), bbox_inches="tight")
plt.close()
print("✓ Figure 4 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Per-group rates before vs after (for best-case detector)
# ─────────────────────────────────────────────────────────────────────────────
# Pick detector with highest SPD before debiasing
worst_det = comparison_df[comparison_df["stage"]=="before"].sort_values("SPD", ascending=False).iloc[0]["detector"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
fig.suptitle(f"Figure 5: Per-Group Anomaly Rates Before vs After Debiasing ({worst_det})",
             fontsize=13, fontweight="bold")

for ax, stage in zip(axes, ["before", "after"]):
    sub = comparison_df[(comparison_df["detector"]==worst_det) & (comparison_df["stage"]==stage)].iloc[0]
    rate_cols = [c for c in comparison_df.columns if c.startswith("rate_")]
    groups = [c.replace("rate_","") for c in rate_cols]
    rates  = [sub[c] for c in rate_cols]
    short  = [g.replace("AfricanAmerican","African\nAmerican") for g in groups]
    bars = ax.bar(short, rates, color=PALETTE_RACE[:len(groups)], edgecolor="white")
    ax.set_title(stage.capitalize(), fontweight="bold")
    ax.set_ylabel("Anomaly Rate" if stage=="before" else "")
    ax.axhline(np.mean(rates), color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"Mean={np.mean(rates):.3f}")
