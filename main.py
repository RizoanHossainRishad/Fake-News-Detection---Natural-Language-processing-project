# main.py


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from definiton_py.preprocessing import preprocess_text
from definiton_py.model_definition import predict


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI()

# -----------------------------
# Request & Response Schemas
# -----------------------------

class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    probability: float


# -----------------------------
# Health Check
# -----------------------------

@app.get("/")
def health_check():
    return {"status": "API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict", response_model=PredictResponse)
def predict_text(request: PredictRequest):
    try:
        # 1. Preprocess text
        vector = preprocess_text(request.text)
        
        # Check if vector is all zeros (common with empty or unknown text)
        if np.all(vector == 0):
             # You can choose to raise an error or return a neutral prediction
             return PredictResponse(label=0, probability=0.5)

        # 2. Model inference - ENSURE this function is called and result used
        result = predict(vector)

        # 3. Explicitly map the dictionary to the response model
        return PredictResponse(
            label=result["label"],
            probability=result["probability"]
        )

    except Exception as e:
        # This helps you see the actual Python error in your logs
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
