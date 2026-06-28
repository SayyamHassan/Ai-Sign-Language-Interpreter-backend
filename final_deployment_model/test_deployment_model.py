import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_bilstm_model.keras"
LABELS_PATH = BASE_DIR / "labels.npy"
CONFIG_PATH = BASE_DIR / "feature_config.json"


# =========================
# LOAD CONFIG
# =========================

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

sequence_length = config["sequence_length"]
feature_size = config["feature_size"]
top_k = config["prediction_output"]["top_k"]

print("Config loaded successfully.")
print("Sequence length:", sequence_length)
print("Feature size:", feature_size)


# =========================
# LOAD LABELS
# =========================

labels = np.load(LABELS_PATH, allow_pickle=True)
labels = np.array([str(label) for label in labels])

print("Labels loaded successfully.")
print("Total labels:", len(labels))
print("First 10 labels:", labels[:10])


# =========================
# LOAD MODEL
# =========================

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

model.summary()


# =========================
# TEST DUMMY INPUT
# =========================

dummy_input = np.zeros(
    (1, sequence_length, feature_size),
    dtype=np.float32
)

print("\nRunning dummy prediction...")
predictions = model.predict(dummy_input, verbose=0)

print("Prediction shape:", predictions.shape)

top_indices = np.argsort(predictions[0])[-top_k:][::-1]

print("\nTop predictions:")
for rank, idx in enumerate(top_indices, start=1):
    print(
        f"{rank}. {labels[idx]} "
        f"(class_index={idx}, confidence={predictions[0][idx]:.6f})"
    )

print("\nDeployment model test completed successfully.")