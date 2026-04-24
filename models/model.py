# models/model.py
import random

GESTURES = ["Wave", "Thumbs Up", "Fist", "Peace", "Okay"]

def predict_gesture(frame_path: str) -> str:
    """
    Dummy function: returns a random gesture for testing.
    Replace with actual MediaPipe / YOLO + trained model later.
    """
    return random.choice(GESTURES)
