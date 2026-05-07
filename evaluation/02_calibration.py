
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

cfg_spec = spec_from_file_location("cfg", Path(__file__).resolve().parent / "00_config.py")
cfg = module_from_spec(cfg_spec); cfg_spec.loader.exec_module(cfg)

EPS = 1e-12


# ---------- proper scoring rules ----------
def brier_binary(p, y):
    return float(np.mean((p - y) ** 2))


def equal_mass_bins(p, y, n_bins=15):
    """Equal-MASS bins (each bin contains ~1/n_bins of the mass)."""
    order = np.argsort(p)
    edges_idx = np.linspace(0, len(p), n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges_idx[i], edges_idx[i + 1]
        if hi <= lo:
            continue
        sl = order[lo:hi]
        rows.append({
            "bin": i,
            "n": int(len(sl)),
            "p_mean": float(p[sl].mean()),
            "p_lo":   float(p[sl].min()),
            "p_hi":   float(p[sl].max()),
            "y_rate": float(y[sl].mean()),
        })
    return pd.DataFrame(rows)


def ece_mce_from_bins(rel: pd.DataFrame, n_total: int):
    gap = (rel.p_mean - rel.y_rate).abs().to_numpy()
    w   = rel.n.to_numpy() / n_total
    ece = float((w * gap).sum())
    mce = float(gap.max() if len(gap) else 0.0)
    return ece, mce


# ---------- post-hoc calibrators ----------
def fit_platt(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs")
    lr.fit(z, y)
    return lr


def apply_platt(lr, p):
    p = np.clip(p, EPS, 1 - EPS)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def apply_isotonic(iso, p):
    return iso.transform(p)


def fit_temperature_seizure(raw_logits_full, y, seizure_idx=0):
    """Optimise a single scalar T over the 6-class softmax, then read off
    the seizure column.  Optimises NLL on the binary task (seizure-vs-rest).
    """
    def nll(T):
        if T <= 0:
            return 1e9
        z = raw_logits_full / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z); e /= e.sum(axis=1, keepdims=True)
        p = np.clip(e[:, seizure_idx], EPS, 1 - EPS)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def apply_temperature_seizure(raw_logits_full, T, seizure_idx=0):
    z = raw_logits_full / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); e /= e.sum(axis=1, keepdims=True)
    return e[:, seizure_idx]


# ---------- recall-pinned threshold helper ----------
def tau_at_recall(p, y, target=0.95):
    pos = p[y == 1]
    if len(pos) == 0:
        return np.nan
    q = np.quantile(pos, 1.0 - target)
    return float(q)


def main():
    df = pd.read_parquet(cfg.DATA / "long_oof_full.parquet")
    rng = np.random.default_rng(cfg.RNG_SEED)

    raw_rows = []
    rel_rows = []
    drift_rows = []

    for model in cfg.MODELS:
        sub = df[df.model == model].copy()
        p_all = sub["p_seizure"].to_numpy()
        y_all = sub["y_true"].to_numpy().astype(int)
        raw_all = sub[[f"raw_{i}" for i in range(6)]].to_numpy()

        # ---- raw calibration ----
        rel = equal_mass_bins(p_all, y_all, cfg.N_BINS_CAL)
        ece, mce = ece_mce_from_bins(rel, len(p_all))
        bri = brier_binary(p_all, y_all)
        raw_rows.append({"model": model, "method": "raw",
                         "ECE": ece, "MCE": mce, "Brier": bri,
                         "mean_p": float(p_all.mean()),
                         "prevalence": float(y_all.mean())})
        rel["model"] = model; rel["method"] = "raw"
        rel_rows.append(rel)

        # ---- LOFO calibration: fit on 4 folds, evaluate on held-out fold ----
        # Then aggregate held-out predictions across all 5 folds and recompute.
        folds = sorted(sub.fold.unique())
        p_platt = np.empty_like(p_all)
        p_iso   = np.empty_like(p_all)
        p_temp  = np.empty_like(p_all)
        # also collect per-fold temperature/T values
        T_per_fold = []
        for f in folds:
            tr = sub.fold != f
            te = sub.fold == f
            p_tr, y_tr = p_all[tr], y_all[tr]
            p_te        = p_all[te]
            raw_tr      = raw_all[tr]
            raw_te      = raw_all[te]

            lr = fit_platt(p_tr, y_tr)
            iso = fit_isotonic(p_tr, y_tr)
            T  = fit_temperature_seizure(raw_tr, y_tr, cfg.SEIZURE_IDX)
            T_per_fold.append({"model": model, "fold": int(f), "T": T})

            p_platt[te] = apply_platt(lr, p_te)
            p_iso[te]   = apply_isotonic(iso, p_te)
            p_temp[te]  = apply_temperature_seizure(raw_te, T, cfg.SEIZURE_IDX)

        for tag, pcal in [("platt", p_platt), ("isotonic", p_iso), ("temperature", p_temp)]:
            rel_c = equal_mass_bins(pcal, y_all, cfg.N_BINS_CAL)
            ece_c, mce_c = ece_mce_from_bins(rel_c, len(pcal))
            bri_c = brier_binary(pcal, y_all)
            raw_rows.append({"model": model, "method": tag,
                             "ECE": ece_c, "MCE": mce_c, "Brier": bri_c,
                             "mean_p": float(pcal.mean()),
                             "prevalence": float(y_all.mean())})
            rel_c["model"] = model; rel_c["method"] = tag
            rel_rows.append(rel_c)

        pd.DataFrame(T_per_fold).to_csv(
            cfg.TABLES / f"temperature_per_fold_{model}.csv", index=False)

        # ---- threshold drift: τ@R=0.95 across folds, raw vs each calibrator ----
        target = 0.95
        for tag, pcal in [("raw", p_all), ("platt", p_platt),
                          ("isotonic", p_iso), ("temperature", p_temp)]:
            taus = []
            for f in folds:
                te = sub.fold == f
                taus.append(tau_at_recall(pcal[te], y_all[te], target))
            taus = np.array(taus)
            mu = float(np.mean(taus)); sd = float(np.std(taus, ddof=1))
            drift_rows.append({
                "model": model, "method": tag,
                "tau_mean": mu, "tau_sd": sd,
                "tau_min": float(taus.min()), "tau_max": float(taus.max()),
                "tau_ratio_max_min": float(taus.max() / max(taus.min(), 1e-9)),
                "tau_cv": float(sd / max(mu, 1e-9)),
                "tau_per_fold": ",".join(f"{t:.5f}" for t in taus),
            })

    # ---- write tables ----
    cal_summary = pd.DataFrame(raw_rows)
    cal_summary.to_csv(cfg.TABLES / "calibration_summary.csv", index=False)
    rel_long = pd.concat(rel_rows, ignore_index=True)
    rel_long.to_csv(cfg.TABLES / "calibration_reliability.csv", index=False)
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(cfg.TABLES / "tau_drift_by_calibration.csv", index=False)

    print("\n=== Calibration summary (lower is better) ===")
    print(cal_summary.pivot_table(index="model", columns="method",
                                  values=["ECE", "MCE", "Brier"]).round(4))
    print("\n=== τ@R=0.95 across folds — does calibration close the drift? ===")
    print(drift_df[["model", "method", "tau_mean", "tau_sd",
                    "tau_ratio_max_min", "tau_cv"]].round(4).to_string(index=False))

    # ---- reliability diagram figure ----
    methods = ["raw", "platt", "isotonic", "temperature"]
    fig, axes = plt.subplots(len(cfg.MODELS), len(methods),
                             figsize=(4 * len(methods), 3.5 * len(cfg.MODELS)),
                             sharex=True, sharey=True)
    for i, model in enumerate(cfg.MODELS):
        for j, m in enumerate(methods):
            ax = axes[i, j] if axes.ndim == 2 else axes[j]
            sub = rel_long[(rel_long.model == model) & (rel_long.method == m)]
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
            sizes = 18 + 600 * (sub.n / sub.n.max())
            ax.scatter(sub.p_mean, sub.y_rate, s=sizes, alpha=0.65,
                       edgecolor="black", linewidth=0.4)
            ax.plot(sub.p_mean, sub.y_rate, lw=1, alpha=0.6)
            row = cal_summary[(cal_summary.model == model) & (cal_summary.method == m)].iloc[0]
            ax.set_title(f"{model} — {m}\nECE={row.ECE:.3f}  Brier={row.Brier:.3f}",
                         fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if j == 0: ax.set_ylabel("Empirical seizure rate")
            if i == len(cfg.MODELS) - 1: ax.set_xlabel("Predicted p(seizure)")
    fig.suptitle("Reliability diagrams — raw vs post-hoc calibrators (held-out per fold)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "02_reliability_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- threshold drift figure ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods_plot = ["raw", "platt", "isotonic", "temperature"]
    width = 0.18
    x = np.arange(len(methods_plot))
    for k, model in enumerate(cfg.MODELS):
        cv = [drift_df[(drift_df.model == model) & (drift_df.method == m)].tau_cv.iloc[0]
              for m in methods_plot]
        ax.bar(x + (k - 1) * width, cv, width, label=model)
    ax.set_xticks(x); ax.set_xticklabels(methods_plot)
    ax.set_ylabel("CV(τ@R=0.95) across 5 folds  (lower = more transferable)")
    ax.set_title("Does post-hoc calibration tighten the τ-drift?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "02_tau_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {cfg.FIGURES}/02_reliability_grid.png and 02_tau_drift.png")


if __name__ == "__main__":
    main()
