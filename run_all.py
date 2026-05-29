"""
run_all.py
----------
Runs the full research pipeline end-to-end.

Usage:
    python run_all.py

Steps:
    1. Preprocessing
    2. Anomaly Detection (IF, LOF, OCSVM, DBSCAN)
    3. Fairness Metrics
    4. Debiasing (FairnessWrapper)
    5. Visualization (all 6 paper figures)
"""

import subprocess
import sys
import time

STEPS = [
    ("01_preprocessing.py",    "Step 1/5: Preprocessing"),
    ("02_anomaly_detection.py","Step 2/5: Anomaly Detection"),
    ("03_fairness_metrics.py", "Step 3/5: Fairness Metrics"),
    ("04_debiasing.py",        "Step 4/5: Debiasing"),
    ("05_visualization.py",    "Step 5/5: Visualization"),
]

print("=" * 60)
print("  Fairness-Aware Anomaly Detection — Full Pipeline")
print("  IEEE ICDM 2026 Research")
print("=" * 60)

total_start = time.time()

for script, label in STEPS:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n✗ {script} failed. Stopping pipeline.")
        sys.exit(1)
    print(f"\n  ✓ Completed in {time.time()-t:.1f}s")

print(f"\n{'='*60}")
print(f"  Pipeline complete in {time.time()-total_start:.1f}s")
print(f"  Results → results/")
print(f"  Figures → figures/")
print(f"{'='*60}")
