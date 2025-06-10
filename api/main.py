import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import PredictionInput, PredictionOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

app = FastAPI(
    title="MLOps Project API",
    description="API for the damage incidence prediction model.",
    version="0.1.0",
)

model_artifacts = {}


@app.on_event("startup")
def load_model():
    logging.info("Loading model artifacts...")
    artifacts_path = Path("ml/artifacts")
    model_artifacts["model"] = joblib.load(artifacts_path / "svc_model.joblib")
    model_artifacts["scaler"] = joblib.load(artifacts_path / "scaler.joblib")
    model_artifacts["numerical_features"] = [
        field_name
        for field_name, field in PredictionInput.__fields__.items()
        if field.annotation in [float, int]
    ]
    logging.info("Artifacts loaded successfully.")


@app.get("/")
def read_root():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    input_df = pd.DataFrame([data.dict()])

    numerical_features = model_artifacts["numerical_features"]
    scaler = model_artifacts["scaler"]
    model = model_artifacts["model"]

    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    prediction_proba = model.predict_proba(input_df)
    prediction = int(prediction_proba.argmax(axis=1)[0])
    probability = float(prediction_proba[0, prediction])

    return PredictionOutput(prediction=prediction, probability=probability)
