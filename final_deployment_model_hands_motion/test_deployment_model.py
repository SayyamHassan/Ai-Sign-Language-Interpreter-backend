import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# CUSTOM POSITION EMBEDDING LAYER
# ============================================================

class PositionEmbedding(layers.Layer):
    """
    Required custom layer for loading the trained Transformer model.
    This must match the PositionEmbedding layer used during training.
    """

    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim
        })
        return config


# ============================================================
# PATHS
# ============================================================

DEPLOYMENT_DIR = Path("/mnt/d/FYP/FYP model/final_deployment_model_hands_motion")

CONFIG_PATH = DEPLOYMENT_DIR / "feature_config.json"
MODEL_PATH = DEPLOYMENT_DIR / "best_hands_motion_transformer_model.keras"
LABELS_PATH = DEPLOYMENT_DIR / "labels.npy"


# ============================================================
# CHECK FILES
# ============================================================

print("Checking deployment files...\n")

required_files = [
    CONFIG_PATH,
    MODEL_PATH,
    LABELS_PATH
]

for file_path in required_files:
    if file_path.exists():
        print("Found:", file_path)
    else:
        raise FileNotFoundError(f"Missing file: {file_path}")


# ============================================================
# LOAD CONFIG
# ============================================================

print("\nLoading feature configuration...")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

SEQUENCE_LENGTH = int(config["sequence_length"])
TOTAL_FEATURE_SIZE = int(config["total_feature_size"])
MODEL_NAME = config["model_name"]

print("Model name:", MODEL_NAME)
print("Sequence length:", SEQUENCE_LENGTH)
print("Total feature size:", TOTAL_FEATURE_SIZE)
print("Expected input shape:", config["input_shape"])


# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading labels...")

labels = np.load(LABELS_PATH, allow_pickle=True)
labels = np.array([str(label) for label in labels])

print("Total labels:", len(labels))
print("First 10 labels:", labels[:10])


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PositionEmbedding": PositionEmbedding
    }
)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# CREATE FAKE TEST INPUT
# ============================================================

print("\nCreating fake test input...")

fake_input = np.zeros(
    shape=(1, SEQUENCE_LENGTH, TOTAL_FEATURE_SIZE),
    dtype=np.float32
)

print("Fake input shape:", fake_input.shape)


# ============================================================
# RUN TEST PREDICTION
# ============================================================

print("\nRunning test prediction...")

prediction = model.predict(fake_input, verbose=0)

print("Prediction shape:", prediction.shape)

predicted_index = int(np.argmax(prediction[0]))
predicted_label = labels[predicted_index]
confidence = float(prediction[0][predicted_index])

top5_indices = np.argsort(prediction[0])[-5:][::-1]

print("\nTop-1 prediction:")
print("Index:", predicted_index)
print("Label:", predicted_label)
print("Confidence:", confidence)

print("\nTop-5 predictions:")
for rank, index in enumerate(top5_indices, start=1):
    print(
        f"{rank}. index={int(index)}, "
        f"label={labels[index]}, "
        f"confidence={float(prediction[0][index]):.6f}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\nDeployment model test completed successfully.")
print("Your model, labels, and feature config are ready for FastAPI integration.")