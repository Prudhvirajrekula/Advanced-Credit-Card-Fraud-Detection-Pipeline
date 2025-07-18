# Credit_Card_Fraud_Detection_System/Inference_Optimization_Extension/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import traceback

# Path to model relative to this script
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../MLFlow_Optuna_Extension/Best_Model/model.pkl")
)

app = FastAPI()
model = None  # Global reference


# Define input schema — adjust according to your model’s expected input
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded from:", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
        raise


@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Adjust input structure as needed based on how model was trained

        features = [[
            transaction.Time, transaction.Amount,
            transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8,
            transaction.V9, transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28
        ]]

        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
def reload_model():
    try:
        load_model()
        return {"status": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
