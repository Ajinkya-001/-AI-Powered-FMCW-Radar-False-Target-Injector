# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load the model
model = load_model("/app/radar_classifier.h5")

class RadarInput(BaseModel):
    signal: list  # Expecting a list of 512 float values

@app.post("/predict")
def predict(data: RadarInput):
    try:
        x = np.array(data.signal).reshape(1, -1)  # (1, 512)
        prediction = model.predict(x)[0][0]
        label = "Spoofed" if prediction >= 0.5 else "Real"
        confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
        return {"label": label, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
