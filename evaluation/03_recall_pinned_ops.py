
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tables"

TARGETS = [0.90, 0.95, 0.99, 1.00]
MODELS = ["EffNetB0", "TCSNet", "VIPEEGNet"]

rows = []
for m in MODELS:
    sw = pd.read_csv(OUT / f"threshold_sweep_{m}.csv")
    for target in TARGETS:
        ok = sw[sw["sensitivity"] >= target]
        if ok.empty:
            continue
        # highest threshold (= fewest alerts) hitting the recall floor
        pick = ok.loc[ok["threshold"].idxmax()].copy()
        pick["model"] = m
        pick["target_recall"] = target
        rows.append(pick)

op = pd.DataFrame(rows)[[
    "model", "target_recall", "threshold",
    "sensitivity", "specificity", "precision", "NPV",
    "F1", "F2", "Youden_J", "NNR",
    "TP", "FP", "FN", "TN", "FP_per_100_neg",
]]
op.to_csv(OUT / "operating_points.csv", index=False)

# Console pretty-print
print(op.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
print(f"\nSaved: {OUT/'operating_points.csv'}")
