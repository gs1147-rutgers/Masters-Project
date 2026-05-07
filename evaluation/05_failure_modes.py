"""
Step 5 — Failure-mode analysis at the chosen operating point.

A model that fails into a related IIIC pattern (LPD/GPD/LRDA/GRDA) is
much safer than one that calls the missed seizure "Other" — the
clinician would still likely flag the related-pattern alert for review.

For each model at recall = 0.95 we:
  1. find the operating threshold,
  2. take all FN EEGs (true seizure but P(Seizure) < τ),
  3. record the model's argmax class for those EEGs,
  4. partition into 'near-miss' (LPD/GPD/LRDA/GRDA) vs 'total miss'
     (Other).

Outputs:
    tables/missed_seizure_argmax.csv  — per-model class breakdown
    tables/missed_seizure_eegs.csv    — per-EEG list of missed seizures
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tables"
df = pd.read_csv(OUT / "long_predictions.csv")
ops = pd.read_csv(OUT / "operating_points.csv")

# Per-EEG full softmax matrix is not in the long table — pull it from npz once.
ROOT = Path(__file__).resolve().parent.parent
src = np.load(ROOT / "analysis" / "oof_per_eeg.npz", allow_pickle=True)
classes = list(src["classes"])
yp_all = {"EffNetB0": src["yp_eff"], "TCSNet": src["yp_tcs"], "VIPEEGNet": src["yp_vip"]}
eeg_ids = src["eeg_ids"]
yhard = src["yhard"]
yt = src["yt"]
NEAR = {"LPD", "GPD", "LRDA", "GRDA"}

rows, miss_rows = [], []
for model in ["EffNetB0", "TCSNet", "VIPEEGNet"]:
    tau = float(ops[(ops.model == model) & (ops.target_recall == 0.95)]["threshold"].iloc[0])
    yp = yp_all[model]
    p_seiz = yp[:, 0]
    is_seizure = (yhard == 0)
    miss = is_seizure & (p_seiz < tau)
    n_seiz = int(is_seizure.sum())
    n_miss = int(miss.sum())

    if n_miss == 0:
        rows.append({"model": model, "threshold": tau, "n_seizure": n_seiz,
                     "n_missed": 0})
        continue

    # What did the model think these were?
    argmax = yp[miss].argmax(axis=1)
    counts = pd.Series([classes[i] for i in argmax]).value_counts().to_dict()
    near = sum(v for k, v in counts.items() if k in NEAR)
    other = counts.get("Other", 0)
    seiz_argmax = counts.get("Seizure", 0)  # below τ but still argmax seizure
    rows.append({
        "model": model, "threshold": tau, "n_seizure": n_seiz, "n_missed": n_miss,
        "argmax_seizure_below_tau": seiz_argmax,
        "argmax_LPD": counts.get("LPD", 0),
        "argmax_GPD": counts.get("GPD", 0),
        "argmax_LRDA": counts.get("LRDA", 0),
        "argmax_GRDA": counts.get("GRDA", 0),
        "argmax_Other": other,
        "near_miss_count": near,
        "near_miss_pct_of_missed": near / n_miss,
        "total_miss_count": other,
        "total_miss_pct_of_missed": other / n_miss,
    })

    # Per-EEG dump
    miss_idx = np.where(miss)[0]
    for i in miss_idx:
        miss_rows.append({
            "model": model, "eeg_id": int(eeg_ids[i]),
            "p_seizure": float(p_seiz[i]),
            "argmax_class": classes[int(yp[i].argmax())],
            "yt_seizure_soft": float(yt[i, 0]),
        })

failure = pd.DataFrame(rows)
failure.to_csv(OUT / "missed_seizure_argmax.csv", index=False)
pd.DataFrame(miss_rows).to_csv(OUT / "missed_seizure_eegs.csv", index=False)

cols = ["model", "n_missed",
        "argmax_LPD", "argmax_GPD", "argmax_LRDA", "argmax_GRDA",
        "argmax_Other",
        "near_miss_pct_of_missed", "total_miss_pct_of_missed"]
print("\nFailure mode at recall=0.95 (each FN's argmax class):")
print(failure[cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# Overlap: which seizures are missed by all three models?
miss_df = pd.DataFrame(miss_rows)
overlap = (miss_df.groupby("eeg_id")["model"].nunique()
           .reset_index(name="n_models_missing"))
print("\nMissed-by overlap at recall=0.95:")
print(overlap["n_models_missing"].value_counts().sort_index())
overlap.to_csv(OUT / "missed_seizure_overlap.csv", index=False)
