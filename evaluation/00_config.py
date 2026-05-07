"""Shared configuration for the PhD-grade evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent

MODELS = {
    "TCSNet":    DATA_ROOT / "tcsnet",
    "VIPEEGNet": DATA_ROOT / "vipeegnet",
    "EffNetB0":  DATA_ROOT / "effnetb0",
}
N_FOLDS = 5

CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
CLASS_COLORS = {
    "Seizure": "#d62728", "LPD": "#1f77b4", "GPD": "#2ca02c",
    "LRDA": "#9467bd", "GRDA": "#ff7f0e", "Other": "#7f7f7f",
}
SEIZURE_IDX = 0

# Recall targets reused from the binary evaluation
RECALL_TARGETS = [0.90, 0.95, 0.99]

# Calibration / DCA / bootstrap settings
N_BINS_CAL = 15            # equal-mass reliability bins
N_BOOTSTRAP = 2000
RNG_SEED = 2024

ARCH_META = {
    "EffNetB0": {
        "backbone": "EfficientNetB0",
        "params_M": 5.33,           # 5,330,571 params for stock EfficientNetB0 (Tan & Le 2019)
        "flops_G": 0.39,            # ~390 MFLOPs at 224x224
        "weights_disk_MB": 49.1,    # measured: fold0.weights.h5
        "framework": "TF / Keras",
        "input": "8-channel spectrogram",
    },
    "VIPEEGNet": {
        "backbone": "EfficientNetV2-B3 (Sunyuri 2-stage)",
        "params_M": 14.4,           # EffV2B3 has ~14M params
        "flops_G": 3.0,              # ~3 GFLOPs at 300x300 (V2B3)
        "weights_disk_MB": None,    # weights/ folder absent in this dump
        "framework": "TF / Keras",
        "input": "spectrogram, 2-stage",
    },
    "TCSNet": {
        "backbone": "TCS-Net (custom temporal-conv + spectral)",
        "params_M": 54.5,           # estimated from 218 MB .pt blob (= 218e6/4 floats)
        "flops_G": None,             # not analytically known without architecture introspection
        "weights_disk_MB": 218.1,   # measured: fold0_best.pt
        "framework": "PyTorch",
        "input": "raw EEG + spectrogram (15+12+6 staged)",
    },
}

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
for d in (TABLES, FIGURES, DATA):
    d.mkdir(exist_ok=True)
