"""
Factorized utility code for the Mental Health Prediction MLOps project.
Contains functions for data processing, embedding, training, and retraining.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

# --- Constants ---
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

FEATURE_COLS = ["Age", "Gender", "Occupation", "Country", "Severity",
                "Consultation_History", "Stress_Level", "Sleep_Hours",
                "Work_Hours", "Physical_Activity_Hours"]

CATEGORICAL_COLS = ["Gender", "Occupation", "Country", "Severity",
                    "Consultation_History", "Stress_Level"]

TARGET_COL = "Mental_Health_Condition"

N_PCA_COMPONENTS = 5  # Number of PCA components for embedding

RETRAIN_THRESHOLD = 10  # Retrain every k new feedback entries


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset."""
    df = pd.read_csv(filepath)
    return df


def preprocess_features(df: pd.DataFrame, label_encoders: dict = None, fit: bool = True):
    """
    Preprocess features: encode categoricals, handle missing values.
    Returns processed feature DataFrame and label_encoders dict.
    """
    df = df.copy()

    # Fill missing Severity for people with no condition
    df["Severity"] = df["Severity"].fillna("None")

    if label_encoders is None:
        label_encoders = {}

    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, label_encoders


def encode_target(series: pd.Series) -> np.ndarray:
    """Encode target: Yes -> 1, No -> 0."""
    return (series == "Yes").astype(int).values


def build_embedding(X: np.ndarray, pca: PCA = None, scaler: StandardScaler = None, fit: bool = True):
    """
    Apply StandardScaler + PCA embedding.
    Returns embedded data, scaler, and pca objects.
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=N_PCA_COMPONENTS)
        X_embedded = pca.fit_transform(X_scaled)
    else:
        X_scaled = scaler.transform(X)
        X_embedded = pca.transform(X_scaled)

    return X_embedded, scaler, pca


def create_ref_data(raw_filepath: str, output_filepath: str):
    """
    Transform the raw dataset into ref_data.csv with PCA-embedded vectors.
    Also saves the label_encoders, scaler, and PCA model.
    """
    df = load_raw_data(raw_filepath)

    # Encode target
    y = encode_target(df[TARGET_COL])

    # Preprocess features
    df_processed, label_encoders = preprocess_features(df, fit=True)
    X = df_processed[FEATURE_COLS].values

    # Build embedding
    X_embedded, scaler, pca = build_embedding(X, fit=True)

    # Create ref_data DataFrame
    pca_columns = [f"pca_{i}" for i in range(N_PCA_COMPONENTS)]
    ref_df = pd.DataFrame(X_embedded, columns=pca_columns)
    ref_df["target"] = y

    # Save ref_data.csv
    ref_df.to_csv(output_filepath, index=False)
    print(f"ref_data.csv saved to {output_filepath} with shape {ref_df.shape}")

    # Save preprocessing artifacts
    save_artifact(label_encoders, "label_encoders.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(pca, "pca.pkl")

    return ref_df, label_encoders, scaler, pca


def train_model(X: np.ndarray, y: np.ndarray, save: bool = True):
    """
    Train a RandomForest classifier and optionally save it.
    Returns the trained model and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    }
    print(f"Model trained - F1: {metrics['f1_score']:.4f}, Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(classification_report(y_test, y_pred))

    if save:
        save_artifact(model, "model.pkl")

    return model, metrics


def retrain_model():
    """
    Retrain the model using ref_data + prod_data.
    Returns the new model and updated artifacts.
    """
    ref_path = os.path.join(DATA_DIR, "ref_data.csv")
    prod_path = os.path.join(DATA_DIR, "prod_data.csv")

    ref_df = pd.read_csv(ref_path)

    if os.path.exists(prod_path):
        prod_df = pd.read_csv(prod_path)
        # Only use rows that have a user_feedback (validated data)
        if "user_feedback" in prod_df.columns:
            prod_df = prod_df[prod_df["user_feedback"].notna()].copy()
            # Use user_feedback as the target for prod data
            prod_df["target"] = prod_df["user_feedback"].astype(int)

        # Select only embedding columns + target
        pca_columns = [c for c in ref_df.columns if c.startswith("pca_")]
        cols_to_use = pca_columns + ["target"]
        combined_df = pd.concat([ref_df[cols_to_use], prod_df[cols_to_use]], ignore_index=True)
    else:
        combined_df = ref_df

    pca_columns = [c for c in combined_df.columns if c.startswith("pca_")]
    X = combined_df[pca_columns].values
    y = combined_df["target"].values

    model, metrics = train_model(X, y, save=True)
    print(f"Model retrained on {len(combined_df)} samples")

    return model, metrics


def save_artifact(obj, filename: str):
    """Save a Python object as a pickle file in the artifacts directory."""
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Artifact saved: {filepath}")


def load_artifact(filename: str):
    """Load a pickle artifact from the artifacts directory."""
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def transform_single_input(data: dict, label_encoders: dict, scaler, pca):
    """
    Transform a single input data point through the full pipeline:
    label encoding -> scaling -> PCA embedding.
    Returns the PCA-embedded vector.
    """
    df = pd.DataFrame([data])

    # Fill missing Severity
    df["Severity"] = df["Severity"].fillna("None")

    # Apply label encoding
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
