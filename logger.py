import csv
import json
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List

from config import PREDICTION_LOG_PATH
log_lock = Lock()




# ============================================================
# CSV LOGGING SETUP
# ============================================================

LOG_COLUMNS = [
    "timestamp",
    "top1_label",
    "top1_index",
    "top1_confidence",
    "top2_label",
    "top2_confidence",
    "top3_label",
    "top3_confidence",
    "top4_label",
    "top4_confidence",
    "top5_label",
    "top5_confidence",
    "top5_json",
    "received_frames",
    "model_input_shape",
    "inference_time_ms"
]


def initialize_prediction_log():
    """
    Creates prediction log CSV with header if it does not exist.
    """

    if not PREDICTION_LOG_PATH.exists():
        with open(PREDICTION_LOG_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def append_prediction_log(
    result: Dict[str, Any],
    received_frames: int,
    model_input_shape: List[int],
    inference_time_ms: float
):
    """
    Appends one prediction record to live_prediction_log.csv.
    This helps tune confidence threshold and stabilizer settings.
    """

    top5 = result.get("top5", [])

    def get_top_label(rank_index: int) -> str:
        if rank_index < len(top5):
            return str(top5[rank_index].get("label", ""))
        return ""

    def get_top_confidence(rank_index: int) -> float:
        if rank_index < len(top5):
            return float(top5[rank_index].get("confidence", 0.0))
        return 0.0

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top1_label": str(result.get("gesture", "")),
        "top1_index": int(result.get("top1", {}).get("index", -1)),
        "top1_confidence": float(result.get("confidence", 0.0)),
        "top2_label": get_top_label(1),
        "top2_confidence": get_top_confidence(1),
        "top3_label": get_top_label(2),
        "top3_confidence": get_top_confidence(2),
        "top4_label": get_top_label(3),
        "top4_confidence": get_top_confidence(3),
        "top5_label": get_top_label(4),
        "top5_confidence": get_top_confidence(4),
        "top5_json": json.dumps(top5, ensure_ascii=False),
        "received_frames": int(received_frames),
        "model_input_shape": json.dumps(model_input_shape),
        "inference_time_ms": round(float(inference_time_ms), 4)
    }

    try:
        with log_lock:
            with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
                writer.writerow(row)
    except OSError as error:
        print(f"[LOG WARNING] Prediction log was skipped: {error}")

def clear_prediction_logs():
    with log_lock:
        with open(PREDICTION_LOG_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
            writer.writeheader()
__all__ = [
    "initialize_prediction_log",
    "append_prediction_log",
    "clear_prediction_logs",
]