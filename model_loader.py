import json
import numpy as np
import tensorflow as tf

from config import (
    CONFIG_PATH,
    MODEL_PATH,
    LABELS_PATH
)

from position_embedding import PositionEmbedding

# ============================================================
# LOAD CONFIG, LABELS, MODEL
# ============================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing feature_config.json: {CONFIG_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

SEQUENCE_LENGTH = int(config["sequence_length"])
COORDINATE_FEATURE_SIZE = int(config["coordinate_feature_size"])
MOTION_FEATURE_SIZE = int(config["motion_feature_size"])
TOTAL_FEATURE_SIZE = int(config["total_feature_size"])

labels = np.load(LABELS_PATH, allow_pickle=True)
labels = np.array([str(label) for label in labels])

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PositionEmbedding": PositionEmbedding
    }
)

print("Hand-Motion Transformer model loaded successfully.")
print("Model path:", MODEL_PATH)
print("Labels:", len(labels))
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
__all__ = [
    "config",
    "model",
    "labels",
    "SEQUENCE_LENGTH",
    "COORDINATE_FEATURE_SIZE",
    "MOTION_FEATURE_SIZE",
    "TOTAL_FEATURE_SIZE",
]