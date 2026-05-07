"""
Step 2 — Threshold sweep across all 1,000 thresholds in [0, 1]
plus the data-driven set of unique p_seizure values.

For each (model, threshold) we record TP/FP/FN/TN and derive:
    Sensitivity (Recall, TPR) = TP / (TP+FN)
    Specificity (TNR)         = TN / (TN+FP)
    FPR                       = 1 - Specificity
    Precision (PPV)           = TP / (TP+FP)
    NPV                       = TN / (TN+FN)
    F1, F2 (recall-weighted), Youden's J = Sens+Spec-1
    NNR (Number Needed to Review) = 1 / Precision
    Alarms_per_hour_proxy = FP / N_negatives  (since these are 50-s
        windows; a clinic-time conversion needs absolute clip count
        which we report as alarms/100 negatives in the CSV).

Outputs:
    tables/threshold_sweep_<model>.csv  — full sweep, one per model
    tables/auc_summary.csv              — AUROC and AUPRC per model
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent / "tables"
df = pd.read_csv(OUT / "long_predictions.csv")

# Unique probability cuts give exact (TP,FP) corners; add a fixed grid for plots.
fixed_grid = np.linspace(0.0, 1.0, 1001)
auc_rows = []
for model, sub in df.groupby("model", sort=False):
    p = sub["p_seizure"].to_numpy()
    y = sub["y_true"].to_numpy().astype(int)
    P = int(y.sum()); N = int(len(y) - P)

    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    auc_rows.append({"model": model, "n": len(y), "n_pos": P, "n_neg": N,
                     "AUROC": auroc, "AUPRC": auprc})

    thresholds = np.unique(np.concatenate([fixed_grid, p]))
    thresholds = np.sort(thresholds)[::-1]      # descending — alerts grow as τ↓

    # Vectorised TP/FP at every threshold via cumulative sums on sorted scores.
    order = np.argsort(-p)
    p_sorted = p[order]; y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    rows = []
    for t in thresholds:
        # Predict positive iff p >= t.  Count how many sorted scores satisfy that.
        k = np.searchsorted(-p_sorted, -t, side="right")
        TP = int(tp_cum[k - 1]) if k > 0 else 0
        FP = int(fp_cum[k - 1]) if k > 0 else 0
        FN = P - TP
        TN = N - FP
        sens = TP / P if P > 0 else 0.0
        spec = TN / N if N > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) and not np.isnan(prec) else np.nan
        f2 = (5 * prec * sens / (4 * prec + sens)) if (prec + sens) and not np.isnan(prec) else np.nan
        nnr = 1 / prec if prec and not np.isnan(prec) and prec > 0 else np.nan
        rows.append((t, TP, FP, FN, TN, sens, spec, 1 - spec, prec, npv,
                     f1, f2, sens + spec - 1, nnr,
                     FP / max(N, 1) * 100))

    out = pd.DataFrame(rows, columns=[
        "threshold", "TP", "FP", "FN", "TN",
        "sensitivity", "specificity", "FPR",
        "precision", "NPV",
        "F1", "F2", "Youden_J", "NNR", "FP_per_100_neg",
    ])
    safe = model.replace("/", "_")
    out.to_csv(OUT / f"threshold_sweep_{safe}.csv", index=False)
    print(f"{model}: {len(out):,} thresholds, AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

pd.DataFrame(auc_rows).to_csv(OUT / "auc_summary.csv", index=False)
print(f"\nSaved AUC summary: {OUT/'auc_summary.csv'}")
