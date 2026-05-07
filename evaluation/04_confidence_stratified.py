
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

cfg_spec = spec_from_file_location("cfg", Path(__file__).resolve().parent / "00_config.py")
cfg = module_from_spec(cfg_spec); cfg_spec.loader.exec_module(cfg)


def main():
    df = pd.read_parquet(cfg.DATA / "long_oof_full.parquet")

    conf_rows = []
    for model in cfg.MODELS:
        sub = df[df.model == model].copy()
        sub["correct"] = (sub.argmax_pred == sub.majority_class).astype(int)
        edges = np.quantile(sub.max_p, np.linspace(0, 1, 11))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        binned = pd.cut(sub.max_p, edges, labels=False, include_lowest=True)
        for b in sorted(binned.dropna().unique()):
            cell = sub[binned == b]
            conf_rows.append({
                "model": model,
                "decile": int(b),
                "n": int(len(cell)),
                "max_p_mean": float(cell.max_p.mean()),
                "accuracy": float(cell.correct.mean()),
                "gap": float(cell.max_p.mean() - cell.correct.mean()),
            })
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(cfg.TABLES / "confidence_stratified.csv", index=False)
    print("=== Confidence-stratified accuracy (decile of max softmax) ===")
    print(conf_df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for model in cfg.MODELS:
        d = conf_df[conf_df.model == model]
        ax.plot(d.max_p_mean, d.accuracy, marker="o", label=model)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("Mean max-softmax in decile  (model confidence)")
    ax.set_ylabel("Top-1 accuracy on majority class")
    ax.set_title("Multi-class confidence vs accuracy\n(below diagonal = overconfidence)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1]
    width = 0.27
    deciles = sorted(conf_df.decile.unique())
    x = np.arange(len(deciles))
    for k, model in enumerate(cfg.MODELS):
        d = conf_df[conf_df.model == model].sort_values("decile")
        ax.bar(x + (k - 1) * width, d.gap.to_numpy(), width, label=model)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"D{i+1}" for i in deciles])
    ax.set_xlabel("Confidence decile")
    ax.set_ylabel("Confidence – accuracy  (>0 = overconfident)")
    ax.set_title("Per-decile overconfidence gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "04_confidence_stratified.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {cfg.FIGURES}/04_confidence_stratified.png")


if __name__ == "__main__":
    main()
