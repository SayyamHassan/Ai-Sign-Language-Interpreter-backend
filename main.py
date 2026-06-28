<<<<<<< HEAD
# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from models.model import predict_gesture
from pydantic import BaseModel
from typing import List
app = FastAPI(title="Gesture Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
=======
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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

BASE_DIR = Path(__file__).resolve().parent

DEPLOYMENT_DIR = BASE_DIR / "final_deployment_model_hands_motion"

CONFIG_PATH = DEPLOYMENT_DIR / "feature_config.json"
MODEL_PATH = DEPLOYMENT_DIR / "best_hands_motion_transformer_model.keras"
LABELS_PATH = DEPLOYMENT_DIR / "labels.npy"

LOG_DIR = BASE_DIR / "prediction_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LOG_PATH = LOG_DIR / "live_prediction_log.csv"

log_lock = Lock()


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
TOP_K = 5

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
print("Prediction log path:", PREDICTION_LOG_PATH)


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


initialize_prediction_log()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Real-Time ASL Hand-Motion Transformer Backend",
    description="FastAPI backend for ASL recognition using hand landmarks + motion features.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    ],
>>>>>>> backend-updates
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
<<<<<<< HEAD
class Landmark(BaseModel):
    x: float
    y: float
    z: float

class HandLandmarks(BaseModel):
    landmarks: List[Landmark]

gesture_history: List[str] = []


@app.post("/api/gestures/detect-gesture")
async def detect_gesture(data: HandLandmarks):
    # data.landmarks is a list of 21 points
    gesture_result = predict_gesture(data.landmarks)
    gesture_history.append(gesture_result)
    return {"gesture": gesture_result}
 
 
#  .\venv\Scripts\Activate    
#  uvicorn main:app --reload  
=======


# ============================================================
# REQUEST MODEL
# ============================================================

class GestureRequest(BaseModel):
    frames: Optional[List[Dict[str, Any]]] = None
    sequence: Optional[List[Dict[str, Any]]] = None


# ============================================================
# FEATURE PROCESSING FUNCTIONS
# ============================================================

def landmark_to_array(landmark: Any) -> List[float]:
    """
    Converts one landmark to [x, y, z].
    Supports:
    - {"x": value, "y": value, "z": value}
    - [x, y, z]
    """

    if isinstance(landmark, dict):
        return [
            float(landmark.get("x", 0.0)),
            float(landmark.get("y", 0.0)),
            float(landmark.get("z", 0.0))
        ]

    if isinstance(landmark, list) or isinstance(landmark, tuple):
        if len(landmark) >= 3:
            return [
                float(landmark[0]),
                float(landmark[1]),
                float(landmark[2])
            ]

    return [0.0, 0.0, 0.0]


def normalize_hand(hand_landmarks: Optional[List[Any]]) -> np.ndarray:
    """
    Normalizes one hand using wrist-centered hand-size normalization.

    Input:
    - 21 hand landmarks

    Output:
    - 63 features: 21 landmarks × 3 coordinates

    Missing hand = zeros.
    """

    if hand_landmarks is None:
        return np.zeros(63, dtype=np.float32)

    if not isinstance(hand_landmarks, list):
        return np.zeros(63, dtype=np.float32)

    if len(hand_landmarks) < 21:
        return np.zeros(63, dtype=np.float32)

    points = np.array(
        [landmark_to_array(point) for point in hand_landmarks[:21]],
        dtype=np.float32
    )

    if points.shape != (21, 3):
        return np.zeros(63, dtype=np.float32)

    wrist = points[0].copy()
    points = points - wrist

    distances = np.linalg.norm(points, axis=1)
    hand_size = float(np.max(distances))

    if hand_size < 1e-6:
        hand_size = 1.0

    points = points / hand_size

    return points.flatten().astype(np.float32)


def extract_frame_features(frame: Dict[str, Any]) -> np.ndarray:
    """
    Creates 126 coordinate features from one frame.

    left hand = 63 features
    right hand = 63 features
    total = 126 features
    """

    left_hand = (
        frame.get("left_hand")
        or frame.get("leftHand")
        or frame.get("left")
        or frame.get("left_landmarks")
    )

    right_hand = (
        frame.get("right_hand")
        or frame.get("rightHand")
        or frame.get("right")
        or frame.get("right_landmarks")
    )

    left_features = normalize_hand(left_hand)
    right_features = normalize_hand(right_hand)

    frame_features = np.concatenate(
        [left_features, right_features],
        axis=0
    )

    return frame_features.astype(np.float32)


def transform_coordinate_sequence(
    coordinate_sequence: np.ndarray,
    swap_hands: bool = False,
    mirror_x: bool = False
) -> np.ndarray:
    """
    Applies live-camera inference variants without changing model format.

    Some webcams/MediaPipe outputs can swap left/right hand labels or mirror
    the x-axis compared with dataset videos. Testing these variants makes the
    live demo more forgiving while still feeding the model 30 x 252 features.
    """

    transformed = coordinate_sequence.copy()

    if swap_hands:
        transformed = np.concatenate(
            [transformed[:, 63:126], transformed[:, 0:63]],
            axis=1
        )

    if mirror_x:
        transformed = transformed.copy()
        transformed[:, 0:63:3] *= -1
        transformed[:, 63:126:3] *= -1

    return transformed.astype(np.float32)


def build_model_input(
    frames: List[Dict[str, Any]],
    swap_hands: bool = False,
    mirror_x: bool = False
) -> np.ndarray:
    """
    Converts 30 frame hand landmarks into model input shape:

    30 × 252

    126 coordinate features + 126 motion features
    """

    if not isinstance(frames, list):
        raise ValueError("frames must be a list.")

    if len(frames) == 0:
        raise ValueError("No frames received.")

    coordinate_sequence = []

    for frame in frames:
        if not isinstance(frame, dict):
            frame = {}

        features = extract_frame_features(frame)
        coordinate_sequence.append(features)

    coordinate_sequence = np.array(coordinate_sequence, dtype=np.float32)

    # If fewer than 30 frames, pad at the beginning with zeros
    if coordinate_sequence.shape[0] < SEQUENCE_LENGTH:
        missing = SEQUENCE_LENGTH - coordinate_sequence.shape[0]
        padding = np.zeros(
            (missing, COORDINATE_FEATURE_SIZE),
            dtype=np.float32
        )
        coordinate_sequence = np.vstack([padding, coordinate_sequence])

    # If more than 30 frames, keep last 30 frames
    if coordinate_sequence.shape[0] > SEQUENCE_LENGTH:
        coordinate_sequence = coordinate_sequence[-SEQUENCE_LENGTH:]

    coordinate_sequence = transform_coordinate_sequence(
        coordinate_sequence,
        swap_hands=swap_hands,
        mirror_x=mirror_x
    )

    motion_sequence = np.zeros_like(coordinate_sequence, dtype=np.float32)
    motion_sequence[1:] = coordinate_sequence[1:] - coordinate_sequence[:-1]

    full_sequence = np.concatenate(
        [coordinate_sequence, motion_sequence],
        axis=1
    )

    if full_sequence.shape != (SEQUENCE_LENGTH, TOTAL_FEATURE_SIZE):
        raise ValueError(
            f"Invalid feature shape: {full_sequence.shape}, "
            f"expected {(SEQUENCE_LENGTH, TOTAL_FEATURE_SIZE)}"
        )

    full_sequence = np.nan_to_num(
        full_sequence,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return np.expand_dims(full_sequence.astype(np.float32), axis=0)


def predict_gesture(model_input: np.ndarray) -> Dict[str, Any]:
    """
    Runs model prediction and returns Top-1 and Top-5 results.
    """

    prediction = model.predict(model_input, verbose=0)[0]

    top1_index = int(np.argmax(prediction))
    top1_label = str(labels[top1_index])
    top1_confidence = float(prediction[top1_index])

    top5_indices = np.argsort(prediction)[-TOP_K:][::-1]

    top5_predictions = []

    for index in top5_indices:
        top5_predictions.append({
            "index": int(index),
            "label": str(labels[index]),
            "confidence": float(prediction[index])
        })

    return {
        "gesture": top1_label,
        "confidence": top1_confidence,
        "top1": {
            "index": top1_index,
            "label": top1_label,
            "confidence": top1_confidence
        },
        "top5": top5_predictions
    }


def predict_best_live_variant(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs prediction on small live-camera variants and returns the strongest one.
    This helps when webcam handedness/mirroring differs from the dataset videos.
    """

    variants = [
        ("original", False, False),
        ("swapped_hands", True, False),
        ("mirrored_x", False, True),
        ("swapped_hands_mirrored_x", True, True)
    ]

    best_result = None
    best_model_input = None

    for variant_name, swap_hands, mirror_x in variants:
        model_input = build_model_input(
            frames,
            swap_hands=swap_hands,
            mirror_x=mirror_x
        )
        result = predict_gesture(model_input)
        result["input_variant"] = variant_name

        if best_result is None or result["confidence"] > best_result["confidence"]:
            best_result = result
            best_model_input = model_input

    return {
        "result": best_result,
        "model_input": best_model_input
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "ASL Hand-Motion Transformer Backend is running.",
        "model": config["model_name"],
        "input_shape": config["input_shape"],
        "labels": int(len(labels)),
        "features": config["features_used"]
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model_name": config["model_name"],
        "model_file": config["model_file"],
        "labels_file": config["labels_file"],
        "num_labels": int(len(labels)),
        "sequence_length": SEQUENCE_LENGTH,
        "coordinate_feature_size": COORDINATE_FEATURE_SIZE,
        "motion_feature_size": MOTION_FEATURE_SIZE,
        "total_feature_size": TOTAL_FEATURE_SIZE,
        "input_shape": config["input_shape"],
        "features_used": config["features_used"],
        "final_test_results": config["final_test_results"],
        "prediction_log_path": str(PREDICTION_LOG_PATH)
    }


@app.post("/api/gestures/detect-gesture")
def detect_gesture(request: GestureRequest):
    try:
        frames = request.frames or request.sequence

        if frames is None:
            raise ValueError(
                "Request must contain 'frames' or 'sequence'."
            )

        start_time = time.time()
        best_prediction = predict_best_live_variant(frames)
        result = best_prediction["result"]
        model_input = best_prediction["model_input"]
        inference_time_ms = (time.time() - start_time) * 1000

        result["received_frames"] = len(frames)
        result["model_input_shape"] = list(model_input.shape)
        result["inference_time_ms"] = round(float(inference_time_ms), 4)

        append_prediction_log(
            result=result,
            received_frames=len(frames),
            model_input_shape=list(model_input.shape),
            inference_time_ms=inference_time_ms
        )

        print(
            f"[PREDICTION] label={result['gesture']} | "
            f"confidence={result['confidence']:.4f} | "
            f"variant={result.get('input_variant', 'original')} | "
            f"time={inference_time_ms:.2f} ms"
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.get("/api/log-info")
def log_info():
    return {
        "prediction_log_path": str(PREDICTION_LOG_PATH),
        "exists": PREDICTION_LOG_PATH.exists()
    }


@app.post("/api/clear-logs")
def clear_logs():
    with log_lock:
        with open(PREDICTION_LOG_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
            writer.writeheader()

    return {
        "message": "Prediction logs cleared successfully.",
        "prediction_log_path": str(PREDICTION_LOG_PATH)
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "labels_loaded": True,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "live_variants_enabled": True,
        "prediction_logging": True,
        "prediction_log_path": str(PREDICTION_LOG_PATH)
    }
>>>>>>> backend-updates
