
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

cfg_spec = spec_from_file_location("cfg", Path(__file__).resolve().parent / "00_config.py")
cfg = module_from_spec(cfg_spec); cfg_spec.loader.exec_module(cfg)

EPS = 1e-12


def soft_ce(p, yt):
    """E_yt[ -log p ] where p, yt are (N, 6) distributions."""
    p = np.clip(p, EPS, 1.0)
    return float(-(yt * np.log(p)).sum(axis=1).mean())


def kl(p, q):
    p = np.clip(p, EPS, 1.0); q = np.clip(q, EPS, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum(axis=1).mean())


def tau_at_recall(p, y, target=0.95):
    pos = p[y == 1]
    if len(pos) == 0:
        return np.nan
    return float(np.quantile(pos, 1.0 - target))


def jaccard(a, b):
    a, b = set(a), set(b)
    if not (a or b): return 1.0
    return len(a & b) / len(a | b)


def main():
    df = pd.read_parquet(cfg.DATA / "long_oof_full.parquet")
    rng = np.random.default_rng(cfg.RNG_SEED)

    # ============================================================
    # A. Soft-label fit quality (global + per fold)
    # ============================================================
    soft_rows = []
    for model in cfg.MODELS:
        sub = df[df.model == model]
        yt = sub[[f"yt_{i}" for i in range(6)]].to_numpy()
        yp = sub[[f"yp_{i}" for i in range(6)]].to_numpy()
        y_hard = sub.majority_class.to_numpy()
        p_hard = yp[np.arange(len(yp)), y_hard]

        soft_rows.append({
            "model": model, "fold": "ALL",
            "CE_hard": float(-np.log(np.clip(p_hard, EPS, 1)).mean()),
            "CE_soft": soft_ce(yp, yt),
            "KL_pred_to_yt": kl(yp, yt),
            "KL_yt_to_pred": kl(yt, yp),
            "n": len(sub),
        })
        for f in sorted(sub.fold.unique()):
            sf = sub[sub.fold == f]
            yt_f = sf[[f"yt_{i}" for i in range(6)]].to_numpy()
            yp_f = sf[[f"yp_{i}" for i in range(6)]].to_numpy()
            y_hf = sf.majority_class.to_numpy()
            p_hf = yp_f[np.arange(len(yp_f)), y_hf]
            soft_rows.append({
                "model": model, "fold": int(f),
                "CE_hard": float(-np.log(np.clip(p_hf, EPS, 1)).mean()),
                "CE_soft": soft_ce(yp_f, yt_f),
                "KL_pred_to_yt": kl(yp_f, yt_f),
                "KL_yt_to_pred": kl(yt_f, yp_f),
                "n": len(sf),
            })
    soft_df = pd.DataFrame(soft_rows)
    soft_df.to_csv(cfg.TABLES / "soft_label_fit.csv", index=False)
    print("=== Soft-label fit (global) ===")
    print(soft_df[soft_df.fold == "ALL"].round(4).to_string(index=False))

    # ============================================================
    # B. Sensitivity stratified by inter-rater agreement
    # ============================================================
    bins = [(0.50, 0.65), (0.65, 0.80), (0.80, 1.0001)]
    stra_rows = []
    for model in cfg.MODELS:
        sub = df[df.model == model].copy()
        p = sub.p_seizure.to_numpy(); y = sub.y_true.to_numpy().astype(int)
        tau = tau_at_recall(p, y, 0.95)
        sub["pred_pos"] = (p >= tau).astype(int)
        seiz = sub[sub.y_true == 1].copy()
        for lo, hi in bins:
            cell = seiz[(seiz.seizure_vote_share >= lo) & (seiz.seizure_vote_share < hi)]
            if len(cell) == 0:
                continue
            stra_rows.append({
                "model": model,
                "vote_share_bin": f"[{lo:.2f},{hi:.2f})",
                "n_cases": int(len(cell)),
                "sensitivity": float(cell.pred_pos.mean()),
                "n_missed": int((cell.pred_pos == 0).sum()),
                "tau_used": tau,
            })
    stra_df = pd.DataFrame(stra_rows)
    stra_df.to_csv(cfg.TABLES / "sensitivity_by_agreement.csv", index=False)
    print("\n=== Sensitivity by inter-rater agreement (R=0.95 op-point) ===")
    print(stra_df.round(4).to_string(index=False))

    # ============================================================
    # C. Are missed seizures the ambiguous ones?  (vote-share comparison)
    # ============================================================
    miss_rows = []
    miss_ids_per_model = {}
    for model in cfg.MODELS:
        sub = df[df.model == model].copy()
        p = sub.p_seizure.to_numpy(); y = sub.y_true.to_numpy().astype(int)
        tau = tau_at_recall(p, y, 0.95)
        seiz = sub[sub.y_true == 1].copy()
        seiz["caught"] = (seiz.p_seizure >= tau).astype(int)
        missed = seiz[seiz.caught == 0]
        caught = seiz[seiz.caught == 1]
        diff = float(caught.seizure_vote_share.mean() - missed.seizure_vote_share.mean())

        # Bootstrap CI on the difference
        boot = []
        for _ in range(cfg.N_BOOTSTRAP):
            sample = seiz.sample(frac=1, replace=True, random_state=rng.integers(1 << 31))
            sm = sample[sample.caught == 0]
            sc = sample[sample.caught == 1]
            if len(sm) > 0 and len(sc) > 0:
                boot.append(sc.seizure_vote_share.mean() - sm.seizure_vote_share.mean())
        boot = np.asarray(boot)
        miss_rows.append({
            "model": model,
            "n_missed": int(len(missed)),
            "n_caught": int(len(caught)),
            "mean_vote_share_missed": float(missed.seizure_vote_share.mean()),
            "mean_vote_share_caught": float(caught.seizure_vote_share.mean()),
            "delta": diff,
            "delta_ci_lo": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
            "delta_ci_hi": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
            "tau": tau,
        })
        miss_ids_per_model[model] = set(zip(missed.fold, missed.eeg_id, missed.spectrogram_id))
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(cfg.TABLES / "missed_vs_ambiguous.csv", index=False)
    print("\n=== Missed seizures vs caught seizures (vote-share Δ) ===")
    print(miss_df.round(4).to_string(index=False))

    # ============================================================
    # D. Cross-model concordance of misses
    # ============================================================
    models = list(cfg.MODELS.keys())
    conc_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            A, B = miss_ids_per_model[a], miss_ids_per_model[b]
            conc_rows.append({
                "pair": f"{a} vs {b}",
                "missed_A": len(A),
                "missed_B": len(B),
                "missed_in_both": len(A & B),
                "missed_in_either": len(A | B),
                "jaccard": jaccard(A, B),
            })
    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(cfg.TABLES / "miss_concordance.csv", index=False)
    print("\n=== Cross-model miss concordance ===")
    print(conc_df.round(4).to_string(index=False))

    # ============================================================
    # Figure: vote-share distribution of missed vs caught seizures
    # ============================================================
    fig, axes = plt.subplots(1, len(cfg.MODELS), figsize=(4 * len(cfg.MODELS), 4),
                             sharey=True)
    for ax, model in zip(axes, cfg.MODELS):
        sub = df[df.model == model]
        p = sub.p_seizure.to_numpy(); y = sub.y_true.to_numpy().astype(int)
        tau = tau_at_recall(p, y, 0.95)
        seiz = sub[sub.y_true == 1].copy()
        seiz["caught"] = (seiz.p_seizure >= tau).astype(int)
        ax.hist([seiz[seiz.caught == 1].seizure_vote_share,
                 seiz[seiz.caught == 0].seizure_vote_share],
                bins=np.linspace(0.5, 1.0, 11),
                label=["caught", "missed"], stacked=False, alpha=0.75,
                color=["#2ca02c", "#d62728"])
        ax.set_title(f"{model}\nmissed n={int((seiz.caught==0).sum())}")
        ax.set_xlabel("seizure_vote_share (expert agreement)")
        ax.legend()
    axes[0].set_ylabel("# cases")
    fig.suptitle("Are missed seizures the ambiguous ones? "
                 "(at recall=0.95)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "03_vote_share_missed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {cfg.FIGURES}/03_vote_share_missed.png")


if __name__ == "__main__":
    main()
