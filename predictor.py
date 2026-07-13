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
    print("\nTOP 5")
    for i in top5_indices:
     print(labels[i], float(prediction[i]))
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
    Confirmed via testing (right=ORANGE, left=BED1) that no swap/mirror
    transform is needed — MediaPipe's Left/Right labels already match
    the training convention. Only the original orientation is used.
    """
    model_input = build_model_input(frames, swap_hands=False, mirror_x=False)
    result = predict_gesture(model_input)
    result["input_variant"] = "original"

    return {
        "result": result,
        "model_input": model_input
    }


__all__ = [
    "predict_best_live_variant",
]