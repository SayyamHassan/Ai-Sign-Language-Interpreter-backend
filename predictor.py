from typing import Dict, List, Any

import numpy as np

from config import TOP_K, FAST_VARIANT_CONFIDENCE
from model_loader import model, labels
from feature_extractor import build_model_input



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
    Runs the normal live-camera prediction first.
    Extra mirrored/hand-swapped variants are only used when the normal result
    is weak, because four model passes per request adds noticeable latency.
    """

    original_model_input = build_model_input(
        frames,
        swap_hands=False,
        mirror_x=False
    )
    original_result = predict_gesture(original_model_input)
    original_result["input_variant"] = "original"

    if original_result["confidence"] >= FAST_VARIANT_CONFIDENCE:
        return {
            "result": original_result,
            "model_input": original_model_input
        }

    variants = [
        ("swapped_hands", True, False),
        ("mirrored_x", False, True),
        ("swapped_hands_mirrored_x", True, True)
    ]

    best_result = original_result
    best_model_input = original_model_input

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

__all__ = [
    "predict_best_live_variant",
]