# Fairness-Aware Anomaly Detection in Clinical Data

**Research Paper:** *Demographic Disparities in Unsupervised Anomaly Detection: A Fairness Analysis with Applications to Clinical Risk Screening*

**Target Venue:** IEEE ICDM 2026 (Main Track) — Abstract deadline May 30, 2026

---

## Research Questions

- **RQ1:** Do standard anomaly detectors produce significantly different anomaly score distributions across demographic groups?
- **RQ2:** Which algorithms exhibit the greatest demographic disparity, and under what conditions?
- **RQ3:** Can a lightweight post-processing debiasing wrapper reduce disparity without sacrificing detection performance?

## Algorithms Evaluated

| Algorithm | Type |
|-----------|------|
| Isolation Forest (IF) | Ensemble / tree-based |
| Local Outlier Factor (LOF) | Density-based |
| One-Class SVM (OCSVM) | Kernel-based |
| DBSCAN | Clustering-based |

## Dataset

**Diabetes 130-US Hospitals (1999–2008)**
- Source: UCI ML Repository (ID: 296)
- 101,766 patient records, 50 features
- Demographics: Race, Gender, Age
- Clinical: labs, medications, diagnoses, readmission

## Fairness Metrics

- **Statistical Parity Difference (SPD):** difference in anomaly rate between groups
- **Disparate Impact Ratio (DIR):** ratio of anomaly rates across groups
- **KL Divergence:** distance between score distributions across groups
- **Score Gap:** mean anomaly score difference between demographic groups

## Debiasing Wrapper

A lightweight, algorithm-agnostic post-processing wrapper (`FairnessWrapper`) that:
1. Computes anomaly scores per group
2. Applies per-group score normalization (z-score within group)
3. Re-ranks anomalies to equalize flagging rates across groups

## Project Structure

```
fairness-anomaly-detection/
├── data/
│   ├── diabetic_data.csv        # Raw dataset
│   └── IDS_mapping.csv          # ID mappings
├── figures/                     # Generated plots
├── results/                     # CSV outputs
├── 01_preprocessing.py          # Data cleaning & feature engineering
├── 02_anomaly_detection.py      # Run all 4 detectors
├── 03_fairness_metrics.py       # Compute disparity metrics
├── 04_debiasing.py              # FairnessWrapper + evaluation
├── 05_visualization.py          # All paper figures
├── run_all.py                   # Run full pipeline end-to-end
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Run full pipeline
python run_all.py

# Or step by step
python 01_preprocessing.py
python 02_anomaly_detection.py
python 03_fairness_metrics.py
python 04_debiasing.py
python 05_visualization.py
```

## Requirements

See `requirements.txt`. Python 3.8+.

---

*This repository is part of ongoing research submitted to IEEE ICDM 2026.*
