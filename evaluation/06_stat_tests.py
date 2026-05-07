"""
Step 6 — Statistical rigour.

(a) DeLong's test for pairwise differences in AUROC, with Z-statistic
    and two-sided p-value.  Implementation follows Sun & Xu (2014) —
    O(N log N) using mid-rank arrays.

(b) McNemar's test at the recall=0.95 operating point of each model.
    Compares paired binary correctness on the same EEGs.  Reported with
    continuity correction.

(c) Bootstrap 95 % CIs on AUROC, AUPRC, recall-pinned precision and FP.
    We do not have patient IDs, so we cluster-bootstrap *by fold* (B=1000
    resamples of the five fold IDs with replacement) — this respects
    the patient-disjoint CV design and is more conservative than naive
    EEG-level resampling.  Limitation noted in the report.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(2024)
OUT = Path(__file__).resolve().parent / "tables"
df = pd.read_csv(OUT / "long_predictions.csv")
ops = pd.read_csv(OUT / "operating_points.csv")

# ---------------------- (a) DeLong's test ----------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fast_delong(predictions_sorted, label_1_count):
    """predictions_sorted: shape (k, N), positives first along axis 1."""
    m, n = label_1_count, predictions_sorted.shape[1] - label_1_count
    pos = predictions_sorted[:, :m]
    neg = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(predictions_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n))
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    delong_cov = sx / m + sy / n
    return aucs, delong_cov

def delong_p(preds_a, preds_b, y):
    order = np.argsort(-y, kind="stable")  # positives first (y descending)
    y_sorted = y[order]
    pa = preds_a[order]; pb = preds_b[order]
    m = int(y_sorted.sum())
    aucs, cov = fast_delong(np.vstack([pa, pb]), m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], diff, z, p

# ----- collect predictions per model -----
preds = {}
y = None
for m, sub in df.groupby("model", sort=False):
    sub = sub.sort_values("eeg_id").reset_index(drop=True)
    if y is None:
        y = sub["y_true"].to_numpy().astype(int)
    preds[m] = sub["p_seizure"].to_numpy()

models = list(preds.keys())
delong_rows = []
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        a, b = models[i], models[j]
        auc_a, auc_b, diff, z, p = delong_p(preds[a], preds[b], y)
        delong_rows.append({"model_A": a, "model_B": b,
                            "AUROC_A": auc_a, "AUROC_B": auc_b,
                            "diff_A_minus_B": diff, "Z": z, "p_value": p})
pd.DataFrame(delong_rows).to_csv(OUT / "delong_tests.csv", index=False)
print("DeLong's pairwise AUROC tests:")
for r in delong_rows:
    print(f"  {r['model_A']:>9} vs {r['model_B']:<9}  "
          f"ΔAUC = {r['diff_A_minus_B']:+.4f}   Z = {r['Z']:+.3f}   "
          f"p = {r['p_value']:.4g}")

# ---------------------- (b) McNemar at S=0.95 ----------------------
def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return np.nan, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p

# build {model: hard predictions at its 0.95 threshold}
hard = {}
for m in models:
    tau = float(ops[(ops.model == m) & (ops.target_recall == 0.95)]["threshold"].iloc[0])
    hard[m] = (preds[m] >= tau).astype(int)

mcnemar_rows = []
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        a, b = models[i], models[j]
        ca = (hard[a] == y).astype(int)
        cb = (hard[b] == y).astype(int)
        b_disc = int(((ca == 1) & (cb == 0)).sum())   # A right, B wrong
        c_disc = int(((ca == 0) & (cb == 1)).sum())
        chi2, p = mcnemar_p(b_disc, c_disc)
        mcnemar_rows.append({"model_A": a, "model_B": b,
                             "A_right_B_wrong": b_disc,
                             "A_wrong_B_right": c_disc,
                             "chi2_cont_corr": chi2, "p_value": p})
pd.DataFrame(mcnemar_rows).to_csv(OUT / "mcnemar_tests.csv", index=False)
print("\nMcNemar's tests at recall=0.95 operating points:")
for r in mcnemar_rows:
    print(f"  {r['model_A']:>9} vs {r['model_B']:<9}  "
          f"b={r['A_right_B_wrong']:>4}  c={r['A_wrong_B_right']:>4}  "
          f"χ²={r['chi2_cont_corr']:.3f}   p={r['p_value']:.4g}")

# ---------------------- (c) Cluster bootstrap ----------------------
B = 1000
folds = sorted(df["fold_id"].unique())

def metrics_at_recall(p, y, target):
    P = int(y.sum())
    if P == 0: return np.nan, np.nan, np.nan, np.nan
    order = np.argsort(-p)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    sens = tp / P
    if not (sens >= target).any():
        return np.nan, np.nan, np.nan, np.nan
    k = int(np.argmax(sens >= target))
    TP = int(tp[k]); FP = int(fp[k])
    prec = TP / (TP + FP) if (TP + FP) else np.nan
    return prec, FP, TP, P - TP

ci_rows = []
# Pre-compute per-fold (preds, y) for each model
fold_data = {}
for m in models:
    sub_m = df[df.model == m].sort_values(["fold_id", "eeg_id"]).reset_index(drop=True)
    fold_data[m] = {f: (sub_m[sub_m.fold_id == f]["p_seizure"].to_numpy(),
                        sub_m[sub_m.fold_id == f]["y_true"].to_numpy().astype(int))
                    for f in folds}

for m in models:
    aurocs, auprcs, precs95, fps95 = [], [], [], []
    for _ in range(B):
        sample = rng.choice(folds, size=len(folds), replace=True)
        ps = np.concatenate([fold_data[m][f][0] for f in sample])
        ys = np.concatenate([fold_data[m][f][1] for f in sample])
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        aurocs.append(roc_auc_score(ys, ps))
        auprcs.append(average_precision_score(ys, ps))
        prec, fp, _, _ = metrics_at_recall(ps, ys, 0.95)
        if not np.isnan(prec):
            precs95.append(prec); fps95.append(fp)
    def ci(arr):
        a = np.asarray(arr)
        return (float(np.mean(a)), float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5)))
    rA = ci(aurocs); rP = ci(auprcs); rPr = ci(precs95); rFp = ci(fps95)
    ci_rows.append({
        "model": m,
        "AUROC_mean": rA[0], "AUROC_lo": rA[1], "AUROC_hi": rA[2],
        "AUPRC_mean": rP[0], "AUPRC_lo": rP[1], "AUPRC_hi": rP[2],
        "Prec@R=0.95_mean": rPr[0], "Prec@R=0.95_lo": rPr[1], "Prec@R=0.95_hi": rPr[2],
        "FP@R=0.95_mean": rFp[0], "FP@R=0.95_lo": rFp[1], "FP@R=0.95_hi": rFp[2],
    })
pd.DataFrame(ci_rows).to_csv(OUT / "bootstrap_ci.csv", index=False)
print("\nFold-cluster bootstrap 95% CIs (B=1000):")
for r in ci_rows:
    print(f"  {r['model']}:  AUROC = {r['AUROC_mean']:.3f} "
          f"[{r['AUROC_lo']:.3f}, {r['AUROC_hi']:.3f}],  "
          f"AUPRC = {r['AUPRC_mean']:.3f} "
          f"[{r['AUPRC_lo']:.3f}, {r['AUPRC_hi']:.3f}],  "
          f"Prec@R0.95 = {r['Prec@R=0.95_mean']:.3f} "
          f"[{r['Prec@R=0.95_lo']:.3f}, {r['Prec@R=0.95_hi']:.3f}]")
