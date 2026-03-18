"""
FastAPI Serving API for Mental Health Prediction Model.
Provides endpoints for prediction, feedback collection, and model retraining.
"""

import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import csv
import threading
from scripts.utils import transform_single_input
from scripts.utils import retrain_model

app = FastAPI(
    title="Mental Health Prediction API",
    description="API de serving pour le modèle de prédiction de santé mentale",
    version="1.0.0"
)

# --- Configuration ---
ARTIFACTS_DIR = "/artifacts"
DATA_DIR = "/data"
RETRAIN_THRESHOLD = 10  # Retrain every k feedbacks


# --- Global model variables (loaded at startup) ---
model = None
scaler = None
target_encoder = None
label_encoder = None
model_lock = threading.Lock()


def load_artifact(filename: str):
    """Load a pickle artifact."""
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_artifact(obj, filename: str):
    """Save a pickle artifact."""
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_all_artifacts():
    """Load all model artifacts into global variables."""
    global model, scaler, label_encoder, target_encoder
    model = load_artifact("model.pkl")
    if type(model).__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    scaler = load_artifact("scaler.pkl")
    label_encoder = load_artifact("label_encoder.pkl")
    target_encoder = load_artifact("target_encoder.pkl")
    print("All artifacts loaded successfully.")


@app.on_event("startup")
def startup_event():
    """Load artifacts on API startup."""
    load_all_artifacts()
    # Initialize prod_data.csv if it doesn't exist
    prod_path = os.path.join(DATA_DIR, "prod_data.csv")
    if not os.path.exists(prod_path):
        header = ["prediction", "user_feedback", "target"]
        with open(prod_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"prod_data.csv initialized at {prod_path}")


# --- Request/Response Models ---
class PredictionInput(BaseModel):
    Q3A: int
    Q10A: int
    Q13A: int
    Q16A: int
    Q26A: int
    Q34A: int
    Q37A: int
    Q38A: int
    age: int
    voted: int
    familysize: int
    education: int
    urban: int
    gender: int
    hand: int
    religion: int
    orientation: int
    race: int
    married: int


class PredictionResponse(BaseModel):
    prediction_proba: list
    prediction_label: str
    proba_label: float
    embedding: list


class FeedbackInput(BaseModel):
    embedding: list
    prediction: int
    user_feedback: int  # classe réelle (entre 0 et 4)

class FeedbackResponse(BaseModel):
    message: str
    total_feedbacks: int
    retrain_triggered: bool


# --- Helper functions ---
def transform_input(data: dict) -> np.ndarray:
    """Transform raw input through the preprocessing pipeline."""
    input = transform_single_input(data, label_encoder, target_encoder)
    return scaler.transform(input)


# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Mental Health Prediction API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    """
    Predict mental health condition from input features.
    Returns prediction, probabilities, and the embedding vector.
    """
    try:
        input_dict = data.model_dump()
        X_transformed = transform_input(input_dict)

        with model_lock:
            prediction = model.predict(X_transformed)[0]
            proba = model.predict_proba(X_transformed)[0]
            labels_predictions = ["None", "Mild", "Moderate", "Severe", "Extremely severe"]
            prediction = int(proba.argmax())
            nom_prediction = labels_predictions[prediction]

        return PredictionResponse(
            prediction_proba=proba.tolist(),
            prediction_label=nom_prediction,
            proba_label=proba[prediction],
            embedding=X_transformed[0].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(data: FeedbackInput):
    """
    Submit user feedback (real label) for a prediction.
    Triggers model retraining every RETRAIN_THRESHOLD feedbacks.
    """
    try:
        prod_path = os.path.join(DATA_DIR, "prod_data.csv")

        # Prepare the row
        column_values = data.embedding
        #TODO : verifier que les colonnes sont dans le bon ordre?
        row = column_values + [data.prediction, data.user_feedback, data.user_feedback]

        # Append to prod_data.csv
        with open(prod_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Count total feedbacks
        prod_df = pd.read_csv(prod_path)
        total_feedbacks = len(prod_df)

        # Check if retrain should be triggered
        retrain_triggered = False
        if total_feedbacks > 0 and total_feedbacks % RETRAIN_THRESHOLD == 0:
            print(f"Retrain threshold reached ({total_feedbacks} feedbacks). Triggering retraining...")
            retrain_model()
            retrain_triggered = True

        return FeedbackResponse(
            message="Feedback recorded successfully",
            total_feedbacks=total_feedbacks,
            retrain_triggered=retrain_triggered
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    """Get information about the current model."""
    prod_path = os.path.join(DATA_DIR, "prod_data.csv")
    total_feedbacks = 0
    if os.path.exists(prod_path):
        try:
            prod_df = pd.read_csv(prod_path)
            total_feedbacks = len(prod_df)
        except Exception:
            pass

    return {
        "model_type": type(model).__name__ if model else None,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "total_feedbacks": total_feedbacks,
        "next_retrain_at": ((total_feedbacks // RETRAIN_THRESHOLD) + 1) * RETRAIN_THRESHOLD
    }


class WebhookFeedbackInput(BaseModel):
    """Feedback from n8n webhook (user clicks link in email)."""
    embedding: list
    prediction: int
    user_feedback: int
    user_email: str = ""


@app.post("/webhook/feedback")
def webhook_feedback(data: WebhookFeedbackInput):
    """
    Receive feedback from n8n AI agent webhook.
    Called when user clicks the validation/correction link in their email.
    """
    feedback_data = FeedbackInput(
        embedding=data.embedding,
        prediction=data.prediction,
        user_feedback=data.user_feedback
    )
    result = submit_feedback(feedback_data)
    return {
        "message": f"Feedback from {data.user_email} recorded.",
        "total_feedbacks": result.total_feedbacks,
        "retrain_triggered": result.retrain_triggered
    }


@app.get("/feedback-form")
def feedback_form(embedding: str, prediction: int, email: str = ""):
    """
    Simple HTML feedback form accessible via link in email.
    The user can confirm or correct the prediction.
    """
    import json
    html = f"""
    <html>
    <head><title>Mental Health Prediction - Feedback</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .btn {{ padding: 15px 30px; margin: 10px; font-size: 18px; cursor: pointer; border: none; border-radius: 8px; color: white; }}
        .btn-confirm {{ background-color: #28a745; }}
        .btn-correct {{ background-color: #dc3545; }}
        h1 {{ color: #333; }}
        .prediction {{ font-size: 24px; padding: 20px; background: #f8f9fa; border-radius: 8px; margin: 20px 0; }}
    </style>
    </head>
    <body>
        <h1>🧠 Mental Health Prediction Feedback</h1>
        <div class="prediction">
            <p><strong>Our model predicted:</strong> {"Mental Health Condition Detected" if prediction == 1 else "No Mental Health Condition"}</p>
        </div>
        <p>Please confirm or correct this prediction:</p>
        <form action="/webhook/feedback" method="post" id="feedbackForm">
            <input type="hidden" name="embedding" value='{embedding}'>
            <input type="hidden" name="prediction" value="{prediction}">
            <input type="hidden" name="user_email" value="{email}">
            <button type="button" class="btn btn-confirm" onclick="submitFeedback({prediction})">
                 Confirm Prediction
            </button>
            <button type="button" class="btn btn-correct" onclick="submitFeedback({1 - prediction})">
                 Correct - It's {"No" if prediction == 1 else "Yes"}
            </button>
        </form>
        <script>
            function submitFeedback(feedback) {{
                fetch('/webhook/feedback', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        embedding: {embedding},
                        prediction: {prediction},
                        user_feedback: feedback,
                        user_email: "{email}"
                    }})
                }})
                .then(r => r.json())
                .then(data => {{
                    document.body.innerHTML = '<h1> Thank you!</h1><p>Your feedback has been recorded.</p><p>' + data.message + '</p>';
                }})
                .catch(err => {{
                    document.body.innerHTML = '<h1> Error</h1><p>' + err + '</p>';
                }});
            }}
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)
