"""
01_preprocessing.py
-------------------
Cleans and preprocesses the Diabetes 130-US dataset for anomaly detection.

Steps:
  - Remove duplicates (keep first encounter per patient)
  - Drop high-missingness columns (weight, payer_code, medical_specialty)
  - Replace '?' with NaN, drop remaining missing rows
  - Filter to main racial groups with sufficient sample size
  - Encode categorical clinical features
  - Standardize numerical features
  - Save: processed_features.csv, demographics.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "diabetic_data.csv"))
print(f"  Raw shape: {df.shape}")

# ── Deduplicate (one encounter per patient) ───────────────────────────────────
df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
print(f"  After dedup: {df.shape}")

# ── Drop high-missingness columns ─────────────────────────────────────────────
drop_cols = ["weight", "payer_code", "medical_specialty",
             "encounter_id", "patient_nbr"]
df = df.drop(columns=drop_cols)

# ── Replace '?' with NaN ──────────────────────────────────────────────────────
df = df.replace("?", np.nan)

# ── Filter demographics ───────────────────────────────────────────────────────
# Keep groups with >500 samples for statistical validity
valid_races = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
df = df[df["race"].isin(valid_races)]
df = df[df["gender"].isin(["Male", "Female"])]

# ── Drop rows with remaining NaN only on key clinical columns ─────────────────
# diag_1/2/3 have small missingness — drop those rows only
df = df.dropna(subset=["diag_1", "diag_2", "diag_3"])
print(f"  After filtering: {df.shape}")
print(f"\n  Race distribution:\n{df['race'].value_counts()}")
print(f"\n  Gender distribution:\n{df['gender'].value_counts()}")

# ── Save demographics separately ──────────────────────────────────────────────
demographics = df[["race", "gender", "age"]].reset_index(drop=True)
demographics.to_csv(os.path.join(RESULTS_DIR, "demographics.csv"), index=False)
print(f"\n  Saved demographics: {demographics.shape}")

# ── Feature engineering ───────────────────────────────────────────────────────
df = df.drop(columns=["race", "gender", "age"])

# Age bracket → ordinal
# (age was already dropped above; we used it only for demographics)

# Medication columns: map to numeric
med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 2, "Ch": 1}
med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"]
for col in med_cols:
    if col in df.columns:
        df[col] = df[col].map(med_map).fillna(0).astype(int)

# Binary categoricals
df["change"] = (df["change"] == "Ch").astype(int)
df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

# Glucose / A1C
glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
df["max_glu_serum"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)
df["A1Cresult"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)

# Readmission target (not used in unsupervised detection, but kept for eval)
readmit_map = {"NO": 0, ">30": 1, "<30": 2}
df["readmitted"] = df["readmitted"].map(readmit_map).fillna(0).astype(int)

# Diagnosis codes → numeric (first 3 chars)
for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = pd.to_numeric(df[col].astype(str).str[:3], errors="coerce").fillna(0)

# Drop remaining string columns
df = df.select_dtypes(include=[np.number])
print(f"\n  Feature matrix shape: {df.shape}")

# ── Standardize ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
feature_df = pd.DataFrame(X_scaled, columns=df.columns)

# ── Save ──────────────────────────────────────────────────────────────────────
feature_df.to_csv(os.path.join(RESULTS_DIR, "processed_features.csv"), index=False)
print(f"\n✓ Saved processed_features.csv  shape={feature_df.shape}")
print("✓ Saved demographics.csv")
print("\nPreprocessing complete.")
