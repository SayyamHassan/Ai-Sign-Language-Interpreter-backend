from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEPLOYMENT_DIR = BASE_DIR / "final_deployment_model_hands_motion"

CONFIG_PATH = DEPLOYMENT_DIR / "feature_config.json"
MODEL_PATH = DEPLOYMENT_DIR / "best_hands_motion_transformer_model.keras"
LABELS_PATH = DEPLOYMENT_DIR / "labels.npy"

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = BASE_DIR / "prediction_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LOG_PATH = LOG_DIR / "live_prediction_log.csv"

# ============================================================
# CONSTANTS
# ============================================================

TOP_K = 5
FAST_VARIANT_CONFIDENCE = 0.80