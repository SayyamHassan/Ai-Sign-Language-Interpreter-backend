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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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