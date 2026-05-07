
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent / "tables"
df = pd.read_csv(OUT / "long_predictions.csv")

TARGETS = [0.90, 0.95, 0.99]

def operating_point(p, y, target):
    """Return τ, sens, spec, prec, FP, FN at the highest τ achieving
    sensitivity >= target."""
    P = int(y.sum()); N = int(len(y) - P)
    if P == 0:
        return [np.nan] * 6
    order = np.argsort(-p)
    p_sorted = p[order]; y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    sens = tp_cum / P
    # smallest k such that sens[k] >= target
    idx = np.argmax(sens >= target) if (sens >= target).any() else len(sens) - 1
    TP = int(tp_cum[idx]); FP = int(fp_cum[idx])
    FN = P - TP; TN = N - FP
    tau = float(p_sorted[idx])
    spec = TN / N if N else np.nan
    prec = TP / (TP + FP) if (TP + FP) else np.nan
    return tau, TP / P, spec, prec, FP, FN

stab_rows, fold_metric_rows = [], []
for model, sub in df.groupby("model", sort=False):
    folds = sorted(sub["fold_id"].unique())
    for f in folds:
        s = sub[sub.fold_id == f]
        p = s["p_seizure"].to_numpy()
        y = s["y_true"].to_numpy().astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fold_metric_rows.append({
            "model": model, "fold": int(f), "n": len(y), "n_pos": int(y.sum()),
            "AUROC": roc_auc_score(y, p),
            "AUPRC": average_precision_score(y, p),
        })
        for t in TARGETS:
            tau, sens, spec, prec, FP, FN = operating_point(p, y, t)
            stab_rows.append({
                "model": model, "fold": int(f), "target_recall": t,
                "threshold": tau, "sens_achieved": sens,
                "specificity": spec, "precision": prec,
                "FP": FP, "FN": FN,
            })

stab = pd.DataFrame(stab_rows)
metr = pd.DataFrame(fold_metric_rows)
stab.to_csv(OUT / "per_fold_operating.csv", index=False)
metr.to_csv(OUT / "per_fold_auc.csv", index=False)

# Summary: threshold drift across folds
summary = (stab.groupby(["model", "target_recall"])
                .agg(thr_mean=("threshold", "mean"),
                     thr_std=("threshold", "std"),
                     thr_min=("threshold", "min"),
                     thr_max=("threshold", "max"),
                     prec_mean=("precision", "mean"),
                     prec_std=("precision", "std"))
                .reset_index())
summary["thr_max_over_min"] = summary["thr_max"] / summary["thr_min"]
summary.to_csv(OUT / "per_fold_threshold_drift.csv", index=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

print("\nPer-fold AUROC/AUPRC summary:")
print(metr.groupby("model")[["AUROC", "AUPRC"]]
          .agg(["mean", "std"])
          .round(4))
