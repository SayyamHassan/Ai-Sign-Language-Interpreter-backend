import time
from typing import Any, Dict, List, Optional
from config import (
    PREDICTION_LOG_PATH,
)
from model_loader import (
    config,
    model,
    labels,
    SEQUENCE_LENGTH,
    COORDINATE_FEATURE_SIZE,
    MOTION_FEATURE_SIZE,
    TOTAL_FEATURE_SIZE,
)
from logger import (
    initialize_prediction_log,
    append_prediction_log,
    clear_prediction_logs,
)
from predictor import predict_best_live_variant
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from grammar_service import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    complete_sentence as complete_grammar_sentence,
    ollama_health
)



initialize_prediction_log()



# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Real-Time ASL Hand-Motion Transformer Backend",
    description=(
        "FastAPI backend for ASL recognition using hand landmarks + motion "
        "features, with a separate gloss-to-English grammar module."
    ),
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ============================================================
# REQUEST MODEL
# ============================================================

class GestureRequest(BaseModel):
    frames: Optional[List[Dict[str, Any]]] = None
    sequence: Optional[List[Dict[str, Any]]] = None

class GrammarRequest(BaseModel):
    """Finalized/accepted gloss sequence sent by the Angular frontend."""

    glosses: Optional[List[str]] = None
    raw_gloss: Optional[str] = None
    use_local_llm: bool = True





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
        "features": config["features_used"],
        "grammar_endpoint": "/api/grammar/complete-sentence",
        "grammar_model": OLLAMA_MODEL
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

@app.post("/api/grammar/complete-sentence")
def complete_sentence(request: GrammarRequest):
    """
    Convert a finalized ASL gloss sequence into a natural English sentence.

    This route is separate from sign recognition. It must be called only after
    the frontend has accepted/stabilized the predicted signs.
    """

    try:
        glosses = request.glosses

        if not glosses and request.raw_gloss:
            glosses = request.raw_gloss.strip().split()

        if not glosses:
            raise ValueError(
                "Request must contain a non-empty 'glosses' list or 'raw_gloss'."
            )

        start_time = time.time()
        result = complete_grammar_sentence(
            glosses=glosses,
            use_local_llm=request.use_local_llm
        )
        result["processing_time_ms"] = round(
            (time.time() - start_time) * 1000,
            2
        )

        print(
            f"[GRAMMAR] raw={result['raw_gloss']} | "
            f"sentence={result['completed_sentence']} | "
            f"method={result['method']} | "
            f"time={result['processing_time_ms']:.2f} ms"
        )

        return result

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/grammar/health")
def grammar_health():
    """Report local Ollama availability without affecting recognition."""

    status = ollama_health()
    status["fallback_available"] = True
    return status


@app.get("/api/log-info")
def log_info():
    return {
        "prediction_log_path": str(PREDICTION_LOG_PATH),
        "exists": PREDICTION_LOG_PATH.exists()
    }


@app.post("/api/clear-logs")
def clear_logs():
    clear_prediction_logs()
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
        "prediction_log_path": str(PREDICTION_LOG_PATH),
        "grammar_module_enabled": True,
        "grammar_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "grammar_fallback_available": True
    }
