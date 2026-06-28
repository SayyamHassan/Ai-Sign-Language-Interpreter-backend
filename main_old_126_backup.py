from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Real-Time ASL Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best_lstm_model.keras"
LABELS_PATH = "extracted_data/labels.npy"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 126

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class FrameLandmarks(BaseModel):
    landmarks: List[Landmark]


class LandmarkSequence(BaseModel):
    frames: List[FrameLandmarks]


def convert_frame_to_126_features(frame: FrameLandmarks):
    points = []

    for landmark in frame.landmarks[:42]:
        points.extend([landmark.x, landmark.y, landmark.z])

    while len(points) < FEATURE_SIZE:
        points.append(0.0)

    if len(points) > FEATURE_SIZE:
        points = points[:FEATURE_SIZE]

    return np.array(points)


@app.get("/")
def home():
    return {
        "message": "Real-Time ASL Interpreter API is running"
    }


@app.post("/api/gestures/detect-gesture")
async def detect_gesture(data: LandmarkSequence):
    frames = data.frames

    if len(frames) == 0:
        return {
            "gesture": "No hand detected",
            "confidence": 0
        }

    sequence = []

    for frame in frames[:SEQUENCE_LENGTH]:
        features = convert_frame_to_126_features(frame)
        sequence.append(features)

    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(sequence[-1])

    sequence = np.array(sequence)
    sequence = np.expand_dims(sequence, axis=0)

    prediction = model.predict(sequence, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index] * 100)

    return {
        "gesture": str(labels[predicted_index]),
        "confidence": round(confidence, 2)
    }