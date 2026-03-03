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

app = FastAPI(
    title="Mental Health Prediction API",
    description="API de serving pour le modèle de prédiction de santé mentale",
    version="1.0.0"
)

# --- Configuration ---
ARTIFACTS_DIR = "/artifacts"
DATA_DIR = "/data"
RETRAIN_THRESHOLD = 10  # Retrain every k feedbacks

# --- Feature definitions ---
FEATURE_COLS = ["Age", "Gender", "Occupation", "Country", "Severity",
                "Consultation_History", "Stress_Level", "Sleep_Hours",
                "Work_Hours", "Physical_Activity_Hours"]

CATEGORICAL_COLS = ["Gender", "Occupation", "Country", "Severity",
                    "Consultation_History", "Stress_Level"]

N_PCA_COMPONENTS = 5

# --- Global model variables (loaded at startup) ---
model = None
scaler = None
pca = None
label_encoders = None
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
    global model, scaler, pca, label_encoders
    model = load_artifact("model.pkl")
    scaler = load_artifact("scaler.pkl")
    pca = load_artifact("pca.pkl")
    label_encoders = load_artifact("label_encoders.pkl")
    print("All artifacts loaded successfully.")


@app.on_event("startup")
def startup_event():
    """Load artifacts on API startup."""
    load_all_artifacts()
    # Initialize prod_data.csv if it doesn't exist
    prod_path = os.path.join(DATA_DIR, "prod_data.csv")
    if not os.path.exists(prod_path):
        pca_columns = [f"pca_{i}" for i in range(N_PCA_COMPONENTS)]
        header = pca_columns + ["prediction", "user_feedback", "target"]
        with open(prod_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"prod_data.csv initialized at {prod_path}")


# --- Request/Response Models ---
class PredictionInput(BaseModel):
    Age: int
    Gender: str
    Occupation: str
    Country: str
    Severity: Optional[str] = "None"
    Consultation_History: str
    Stress_Level: str
    Sleep_Hours: float
    Work_Hours: float
    Physical_Activity_Hours: int


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_no: float
    probability_yes: float
    embedding: list


class FeedbackInput(BaseModel):
    embedding: list
    prediction: int
    user_feedback: int  # 0 = No, 1 = Yes (real label from user)


class FeedbackResponse(BaseModel):
    message: str
    total_feedbacks: int
    retrain_triggered: bool


# --- Helper functions ---
def transform_input(data: dict) -> np.ndarray:
    """Transform raw input through the preprocessing pipeline."""
    df = pd.DataFrame([data])
    df["Severity"] = df["Severity"].fillna("None")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    X_embedded = pca.transform(X_scaled)
    return X_embedded


def retrain_model():
    """Retrain the model on ref_data + prod_data and update artifacts."""
    global model

    ref_path = os.path.join(DATA_DIR, "ref_data.csv")
    prod_path = os.path.join(DATA_DIR, "prod_data.csv")

    ref_df = pd.read_csv(ref_path)
    prod_df = pd.read_csv(prod_path)

    # Only use rows with user_feedback
    prod_df = prod_df[prod_df["user_feedback"].notna()].copy()
    prod_df["target"] = prod_df["user_feedback"].astype(int)

    pca_columns = [c for c in ref_df.columns if c.startswith("pca_")]
    cols_to_use = pca_columns + ["target"]

    combined_df = pd.concat([ref_df[cols_to_use], prod_df[cols_to_use]], ignore_index=True)

    X = combined_df[pca_columns].values
    y = combined_df["target"].values

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    new_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    new_model.fit(X, y)

    # Save new model artifact
    save_artifact(new_model, "model.pkl")

    # Update global model
    with model_lock:
        model = new_model

    print(f"Model retrained on {len(combined_df)} samples and deployed.")


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
        X_embedded = transform_input(input_dict)

        with model_lock:
            prediction = int(model.predict(X_embedded)[0])
            proba = model.predict_proba(X_embedded)[0]

        return PredictionResponse(
            prediction=prediction,
            prediction_label="Yes" if prediction == 1 else "No",
            probability_no=float(proba[0]),
            probability_yes=float(proba[1]),
            embedding=X_embedded[0].tolist()
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
        pca_values = data.embedding
        row = pca_values + [data.prediction, data.user_feedback, data.user_feedback]

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
        "n_pca_components": N_PCA_COMPONENTS,
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
                ✅ Confirm Prediction
            </button>
            <button type="button" class="btn btn-correct" onclick="submitFeedback({1 - prediction})">
                ❌ Correct - It's {"No" if prediction == 1 else "Yes"}
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
                    document.body.innerHTML = '<h1>✅ Thank you!</h1><p>Your feedback has been recorded.</p><p>' + data.message + '</p>';
                }})
                .catch(err => {{
                    document.body.innerHTML = '<h1>❌ Error</h1><p>' + err + '</p>';
                }});
            }}
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)
