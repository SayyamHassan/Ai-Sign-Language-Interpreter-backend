from typing import Any, Dict, List, Optional

import numpy as np

from model_loader import (
    SEQUENCE_LENGTH,
    COORDINATE_FEATURE_SIZE,
    TOTAL_FEATURE_SIZE,
)
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
__all__ = [
    "build_model_input",
]