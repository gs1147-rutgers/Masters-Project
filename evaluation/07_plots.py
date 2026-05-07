
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"; OUT.mkdir(parents=True, exist_ok=True)
TBL = ROOT / "tables"
df = pd.read_csv(TBL / "long_predictions.csv")
ops = pd.read_csv(TBL / "operating_points.csv")
auc = pd.read_csv(TBL / "auc_summary.csv")
fold_metr = pd.read_csv(TBL / "per_fold_auc.csv")
fold_op = pd.read_csv(TBL / "per_fold_operating.csv")
miss = pd.read_csv(TBL / "missed_seizure_argmax.csv")

COLORS = {"EffNetB0": "#1f77b4", "TCSNet": "#ff7f0e", "VIPEEGNet": "#2ca02c"}
MODELS = ["EffNetB0", "TCSNet", "VIPEEGNet"]
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 11})

# -------- helpers ----------
def mp(name):
    sub = df[df.model == name]
    return sub["p_seizure"].to_numpy(), sub["y_true"].to_numpy().astype(int)

# ============== fig01 ROC ==============
fig, ax = plt.subplots(figsize=(7, 6))
for m in MODELS:
    p, y = mp(m)
    fpr, tpr, _ = roc_curve(y, p)
    a = float(auc[auc.model == m]["AUROC"].iloc[0])
    ax.plot(fpr, tpr, color=COLORS[m], lw=2, label=f"{m}  AUROC = {a:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate (1 − Specificity)")
ax.set_ylabel("Sensitivity (Recall)")
ax.set_title("ROC — Seizure detection (5-fold OOF, n = 5,939 EEGs)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig01_roc.png"); plt.close(fig)

# ============== fig09 ROC zoom (high-sens corner) ==============
fig, ax = plt.subplots(figsize=(7, 6))
for m in MODELS:
    p, y = mp(m)
    fpr, tpr, _ = roc_curve(y, p)
    a = float(auc[auc.model == m]["AUROC"].iloc[0])
    ax.plot(fpr, tpr, color=COLORS[m], lw=2, label=f"{m}  AUROC = {a:.3f}")
for tgt in (0.90, 0.95, 0.99):
    ax.axhline(tgt, color="gray", lw=0.7, ls=":")
    ax.text(0.55, tgt + 0.005, f"Sens = {tgt:.2f}", color="gray", fontsize=9)
ax.set_xscale("log"); ax.set_xlim(1e-3, 1)
ax.set_ylim(0.5, 1.005)
ax.set_xlabel("False Positive Rate (log scale)")
ax.set_ylabel("Sensitivity (Recall)")
ax.set_title("ROC zoom — high-sensitivity operating region")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig09_roc_zoom.png"); plt.close(fig)

# ============== fig02 PR (with threshold annotations at R=0.90 and R=0.95) ==============
fig, ax = plt.subplots(figsize=(8, 6))
recall_pivots = [0.90, 0.95]
marker_shapes = {0.90: "^", 0.95: "D"}

for m in MODELS:
    p, y = mp(m)
    prec, rec, thr = precision_recall_curve(y, p)
    # precision_recall_curve returns thresholds of length n-1; pad so indices line up
    thr_padded = np.append(thr, 1.0)
    a = float(auc[auc.model == m]["AUPRC"].iloc[0])
    ax.plot(rec, prec, color=COLORS[m], lw=2, label=f"{m}  AUPRC = {a:.3f}")

    for rp in recall_pivots:
        valid = np.where(rec >= rp)[0]
        if len(valid) == 0:
            continue
        idx = valid[-1]      # last index where recall is still ≥ rp
        r_pt, p_pt, t_pt = rec[idx], prec[idx], thr_padded[idx]
        ax.scatter(r_pt, p_pt, marker=marker_shapes[rp], s=80,
                   color=COLORS[m], edgecolor="black", linewidth=0.7, zorder=4)
        offset_y = {0.90: 0.025, 0.95: -0.04}[rp]
        ax.annotate(f"τ={t_pt:.3f}", (r_pt, p_pt),
                    xytext=(r_pt + 0.012, p_pt + offset_y),
                    fontsize=9, color=COLORS[m], ha="left")

prev = float(df[df.model == MODELS[0]]["y_true"].mean())
ax.axhline(prev, color="k", ls="--", lw=1, alpha=0.5,
           label=f"Prevalence = {prev:.3f}")

marker_handles = [plt.Line2D([], [], marker=marker_shapes[rp], color="grey",
                             ls="", markersize=9, markeredgecolor="black",
                             markeredgewidth=0.7, label=f"R = {rp:.2f}")
                  for rp in recall_pivots]
leg1 = ax.legend(loc="upper right", title="Models")
ax.legend(handles=marker_handles, loc="lower left",
          title="Recall pivot (with τ shown)", fontsize=9)
ax.add_artist(leg1)

ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision–Recall — Seizure  (τ shown at R=0.90 and R=0.95)")
fig.tight_layout(); fig.savefig(OUT / "fig02_pr.png"); plt.close(fig)

# ============== fig03 DET ==============
from scipy.stats import norm
def probit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return norm.ppf(x)
fig, ax = plt.subplots(figsize=(7, 6))
for m in MODELS:
    p, y = mp(m)
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    keep = (fpr > 0) & (fnr > 0)
    ax.plot(probit(fpr[keep]), probit(fnr[keep]),
            color=COLORS[m], lw=2, label=m)
ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
ax.set_xticks([probit(t) for t in ticks]); ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.set_yticks([probit(t) for t in ticks]); ax.set_yticklabels([f"{t:g}" for t in ticks])
ax.set_xlim(probit(1e-3), probit(0.5))
ax.set_ylim(probit(1e-3), probit(0.5))
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("False Negative Rate (Missed seizure rate)")
ax.set_title("DET — closer to bottom-left is better")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig03_det.png"); plt.close(fig)

# ============== fig04 threshold overlay (3 panels) ==============
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, m in zip(axes, MODELS):
    sw = pd.read_csv(TBL / f"threshold_sweep_{m}.csv").sort_values("threshold")
    ax.plot(sw.threshold, sw.sensitivity, color="C3", lw=2, label="Sensitivity")
    ax.plot(sw.threshold, sw.specificity, color="C0", lw=2, label="Specificity")
    # Precision and F2 are undefined when the model predicts zero positives;
    # leave NaN so the line breaks rather than dropping spuriously to 0.
    ax.plot(sw.threshold, sw.precision, color="C2", lw=2, label="Precision")
    ax.plot(sw.threshold, sw.F2,        color="C5", lw=1.5, ls=":", label="F2")
    tau95 = float(ops[(ops.model == m) & (ops.target_recall == 0.95)]["threshold"].iloc[0])
    ax.axvline(tau95, color="k", ls="--", lw=1)
    ax.text(tau95, 0.05, f" τ@R=.95\n {tau95:.3f}", fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1)
    ax.set_title(m); ax.set_xlabel("Threshold τ")
axes[0].set_ylabel("Metric value")
axes[0].legend(loc="center left")
fig.suptitle("Sensitivity / Specificity / Precision / F2 vs threshold")
fig.tight_layout(); fig.savefig(OUT / "fig04_threshold_overlay.png"); plt.close(fig)

# ============== fig05 recall-pinned cost bars ==============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
metr_to_plot = [("FP", "False alarms"), ("NNR", "Alerts per true seizure (NNR)")]
for ax, (col, ylabel) in zip(axes, metr_to_plot):
    sub = ops[ops.target_recall.isin([0.95, 0.99])]
    pivot = sub.pivot(index="model", columns="target_recall", values=col).loc[MODELS]
    x = np.arange(len(MODELS)); w = 0.38
    ax.bar(x - w/2, pivot[0.95], w, label="Recall = 0.95", color="#4f9aff")
    ax.bar(x + w/2, pivot[0.99], w, label="Recall = 0.99", color="#ff7f0e")
    for i, m in enumerate(MODELS):
        ax.text(i - w/2, pivot[0.95][m], f"{pivot[0.95][m]:.0f}" if col=="FP" else f"{pivot[0.95][m]:.1f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, pivot[0.99][m], f"{pivot[0.99][m]:.0f}" if col=="FP" else f"{pivot[0.99][m]:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel(ylabel); ax.set_title(ylabel)
    ax.legend()
fig.suptitle("Cost of meeting clinical recall targets")
fig.tight_layout(); fig.savefig(OUT / "fig05_recall_pinned_bars.png"); plt.close(fig)

# ============== fig06 per-fold boxplots ==============
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
    data = [fold_metr[fold_metr.model == m][metric].values for m in MODELS]
    bp = ax.boxplot(data, labels=MODELS, patch_artist=True, widths=0.55)
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.55)
    for i, vals in enumerate(data):
        ax.scatter(np.full_like(vals, i + 1, dtype=float) + 0.05*np.random.randn(len(vals)),
                   vals, color="k", s=18, zorder=3)
    ax.set_title(metric); ax.set_ylabel(metric)
fig.suptitle("Per-fold AUROC / AUPRC (5 folds)")
fig.tight_layout(); fig.savefig(OUT / "fig06_per_fold_box.png"); plt.close(fig)

# ============== fig07 threshold drift ==============
fig, ax = plt.subplots(figsize=(8, 5))
for i, m in enumerate(MODELS):
    sub = fold_op[fold_op.model == m]
    for tgt, marker in zip([0.90, 0.95, 0.99], ["o", "s", "D"]):
        vals = sub[sub.target_recall == tgt]["threshold"].values
        x = np.full_like(vals, i, dtype=float) + (tgt - 0.95) * 4
        ax.scatter(x, vals, marker=marker, s=80, color=COLORS[m],
                   edgecolor="k", lw=0.6,
                   label=(f"{m}  R={tgt}" if i == 0 or True else None))
ax.set_yscale("log")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(MODELS)
ax.set_ylabel("Per-fold operating threshold τ (log scale)")
ax.set_title("Threshold drift across 5 folds (○=R0.90, □=R0.95, ◇=R0.99)")
handles, labels = ax.get_legend_handles_labels()
seen = set(); uniq = []
for h, l in zip(handles, labels):
    if l not in seen: uniq.append((h, l)); seen.add(l)
ax.legend([u[0] for u in uniq[:9]], [u[1] for u in uniq[:9]],
          fontsize=8, ncol=3, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig07_threshold_drift.png"); plt.close(fig)

# ============== fig08 hard misses vs threshold (dynamic) ===================
# How many true seizures does each model miss as we sweep the decision
# threshold τ?  This makes the calibration / score-distribution shift
# directly visible: at the same τ, the three models miss very different
# numbers of seizures.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={"width_ratios": [1.6, 1]})

# --- left panel: full sweep (log-x) ---
ax = axes[0]
n_seiz_total = None
ref_taus = [0.01, 0.05, 0.10, 0.20, 0.50]
table_rows = []  # for the right-panel table

for m in MODELS:
    sw = pd.read_csv(TBL / f"threshold_sweep_{m}.csv").sort_values("threshold")
    if n_seiz_total is None:
        n_seiz_total = int(sw.iloc[0].FN + sw.iloc[0].TP) if len(sw) else 348
        # robust: use the row at threshold = 0 (everything classified positive)
        zero_row = sw[sw.threshold <= 1e-6]
        if len(zero_row):
            n_seiz_total = int(zero_row.iloc[0].TP + zero_row.iloc[0].FN)

    ax.plot(sw.threshold, sw.FN, color=COLORS[m], lw=2.0, label=m)

    # mark this model's τ@R=0.95
    tau95 = float(ops[(ops.model == m) & (ops.target_recall == 0.95)]["threshold"].iloc[0])
    fn95  = int(ops[(ops.model == m) & (ops.target_recall == 0.95)]["FN"].iloc[0])
    ax.scatter([tau95], [fn95], marker="D", s=85, color=COLORS[m],
               edgecolor="black", linewidth=0.7, zorder=5)
    ax.annotate(f"τ@R=0.95\n={tau95:.3f}",
                (tau95, fn95), xytext=(8, -22),
                textcoords="offset points", fontsize=8.5,
                color=COLORS[m])

    # collect missed counts at the canonical reference thresholds
    for t in ref_taus:
        # nearest-row lookup
        idx = (sw.threshold - t).abs().idxmin()
        table_rows.append({
            "model": m, "threshold": t,
            "missed (FN)": int(sw.loc[idx].FN),
            "missed_pct": sw.loc[idx].FN / n_seiz_total,
        })

# horizontal "contract" lines
for tgt, fn_target, lbl in [(0.90, int(round(0.10*n_seiz_total)), "R=0.90"),
                            (0.95, int(round(0.05*n_seiz_total)), "R=0.95"),
                            (0.99, int(round(0.01*n_seiz_total)), "R=0.99")]:
    ax.axhline(fn_target, ls=":", color="grey", lw=0.9, alpha=0.7)
    ax.text(0.0011, fn_target + n_seiz_total * 0.012, f"{lbl} contract  ({fn_target} miss)",
            color="grey", fontsize=8.5)

# vertical reference threshold lines
for t in ref_taus:
    ax.axvline(t, ls=":", color="black", alpha=0.18, lw=0.8)
    ax.text(t, n_seiz_total * 1.02, f"τ={t}",
            ha="center", va="bottom", fontsize=8, color="black", alpha=0.6)

ax.set_xscale("log")
ax.set_xlim(1e-3, 1.0)
ax.set_ylim(0, n_seiz_total * 1.06)
ax.set_xlabel("Decision threshold τ  (log scale)")
ax.set_ylabel(f"Hard misses — true seizures with p_seizure < τ\n(out of n = {n_seiz_total})")
ax.set_title("How does the hard-miss count change with the threshold?")
ax.legend(loc="upper left", title="Model")
ax.grid(True, which="both", alpha=0.3)

# --- right panel: tabular snapshot at canonical thresholds ---
ax = axes[1]
ax.axis("off")
tbl = pd.DataFrame(table_rows)
pivot = tbl.pivot(index="threshold", columns="model", values="missed (FN)")[MODELS]
# build a matplotlib table
header = ["τ"] + MODELS
cell_text = []
for t in ref_taus:
    row = [f"{t:g}"]
    for m in MODELS:
        v = int(pivot.loc[t, m])
        row.append(f"{v}  ({v/n_seiz_total:.0%})")
    cell_text.append(row)
the_table = ax.table(cellText=cell_text, colLabels=header,
                     loc="center", cellLoc="center")
the_table.auto_set_font_size(False); the_table.set_fontsize(10)
the_table.scale(1, 1.7)
for k, m in enumerate(MODELS):
    for r in range(len(ref_taus) + 1):
        cell = the_table[(r, k + 1)]
        if r == 0:
            cell.set_facecolor(COLORS[m]); cell.set_text_props(color="white",
                                                                fontweight="bold")
ax.set_title("Hard-miss count at canonical thresholds\n(absolute count and % of true seizures)",
             fontsize=11, pad=15)

fig.tight_layout(); fig.savefig(OUT / "fig08_failure_modes.png"); plt.close(fig)

print("Saved 9 figures →", OUT)
