from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ANN-based Customer Churn Prediction",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = load_model("model.keras")

scaler = joblib.load("scaler.pkl")


# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerData(BaseModel):

    tenure: float

    MonthlyCharges: float

    TotalCharges: float

    SeniorCitizen: int


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Create DataFrame
    data = pd.DataFrame([{
        "tenure": customer.tenure,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,
        "SeniorCitizen": customer.SeniorCitizen
    }])


    # IMPORTANT:
    # Use the same scaler used during training
    scaled_data = scaler.transform(data)


    # ANN prediction
    probability = model.predict(
        scaled_data,
        verbose=0
    )[0][0]


    # Convert probability to class
    prediction = int(probability >= 0.5)


    # Result
    if prediction == 1:

        result = "Customer is likely to CHURN"

    else:

        result = "Customer is likely to STAY"


    return {

        "churn_probability": round(
            float(probability) * 100,
            2
        ),

        "prediction": prediction,

        "result": result
    }