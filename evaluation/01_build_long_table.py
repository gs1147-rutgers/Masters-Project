"""
Step 1 — Build the long-form table for clinical seizure evaluation.

One row per (model, EEG). Columns:
    model       — 'EffNetB0' | 'TCSNet' | 'VIPEEGNet'
    eeg_id      — int, unique EEG identifier
    fold_id     — int 0..4, patient-disjoint CV fold
    p_seizure   — model's predicted P(Seizure) (softmax column 0)
    y_true      — 1 if expert majority (yhard) is Seizure, else 0
    yt_seizure  — soft label P(Seizure) from expert votes
    yhard_alt   — argmax of yt for missed-seizure failure-mode analysis
                   (0=Seizure, 1=LPD, 2=GPD, 3=LRDA, 4=GRDA, 5=Other)

Source: analysis/oof_per_eeg.npz (already aggregates the 5 folds and
aligns the three models on the shared 5,939 HQ EEGs).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "analysis" / "oof_per_eeg.npz"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(SRC, allow_pickle=True)
classes = list(d["classes"])
SEIZ = classes.index("Seizure")           # 0
y_hard = d["yhard"].astype(int)           # argmax of yt per EEG
y_true = (y_hard == SEIZ).astype(int)     # binary clinical label
yt_s = d["yt"][:, SEIZ].astype(float)     # soft seizure fraction
fold = d["fold"].astype(int)
eeg_ids = d["eeg_ids"].astype(int)

models = {"EffNetB0": d["yp_eff"], "TCSNet": d["yp_tcs"], "VIPEEGNet": d["yp_vip"]}

frames = []
for name, yp in models.items():
    frames.append(pd.DataFrame({
        "model": name,
        "eeg_id": eeg_ids,
        "fold_id": fold,
        "p_seizure": yp[:, SEIZ].astype(float),
        "y_true": y_true,
        "yt_seizure": yt_s,
        "yhard_alt": y_hard,
    }))

long_df = pd.concat(frames, ignore_index=True)
long_df.to_csv(OUT / "long_predictions.csv", index=False)

# Sanity print
print(f"Saved: {OUT/'long_predictions.csv'} ({len(long_df):,} rows)")
print(f"Per model: n_eeg={len(eeg_ids):,}, n_seizure={int(y_true.sum())}, "
      f"prevalence={y_true.mean():.4f}")
print("Per fold (EEGs / seizure positives):")
for f in sorted(np.unique(fold)):
    m = fold == f
    print(f"  fold {f}: n={m.sum():>5}, seizures={int(y_true[m].sum()):>3}")
print("Class names:", classes)
