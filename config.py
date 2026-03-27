from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR       = ROOT / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODEL_DIR      = ROOT / "models"
LOG_DIR        = ROOT / "logs"

for d in [RAW_DATA_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "auc",
    "tree_method": "hist",
}