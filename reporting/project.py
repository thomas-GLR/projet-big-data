"""
Evidently AI Reporting Project.
Generates model performance and data drift reports for the Mental Health Prediction model.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.ui.workspace import Workspace, Project

# --- Configuration ---
DATA_DIR = os.environ.get("DATA_DIR", "/data")
WORKSPACE_PATH = "evidently_workspace"
PROJECT_NAME = "Mental Health Prediction Monitoring"
PROJECT_DESCRIPTION = "Monitoring data drift and model performance for mental health condition prediction"

# PCA column names
N_PCA_COMPONENTS = 5
PCA_COLUMNS = [f"pca_{i}" for i in range(N_PCA_COMPONENTS)]

# Column mapping for Evidently
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=PCA_COLUMNS,
)


def load_reference_data() -> pd.DataFrame:
    """Load reference data (ref_data.csv)."""
    ref_path = os.path.join(DATA_DIR, "ref_data.csv")
    df = pd.read_csv(ref_path)
    # Evidently needs a 'prediction' column for classification metrics
    # For reference data, use target as prediction (perfect baseline)
    if "prediction" not in df.columns:
        df["prediction"] = df["target"]
    return df


def load_production_data() -> pd.DataFrame:
    """Load production data (prod_data.csv)."""
    prod_path = os.path.join(DATA_DIR, "prod_data.csv")
    if not os.path.exists(prod_path):
        print("prod_data.csv not found. Creating synthetic production data for demo.")
        return create_demo_prod_data()

    df = pd.read_csv(prod_path)
    if len(df) < 5:
        print(f"prod_data.csv has only {len(df)} rows. Creating synthetic data for demo.")
        return create_demo_prod_data()

    # Ensure required columns
    if "user_feedback" in df.columns:
        df["target"] = df["user_feedback"].astype(int)

    return df


def create_demo_prod_data() -> pd.DataFrame:
    """Create synthetic production data for initial reporting demo."""
    ref_path = os.path.join(DATA_DIR, "ref_data.csv")
    ref_df = pd.read_csv(ref_path)

    # Take a sample and add some noise to simulate drift
    np.random.seed(42)
    sample_size = min(500, len(ref_df))
    prod_df = ref_df.sample(n=sample_size, random_state=42).copy()

    # Add slight noise to simulate real production data
    for col in PCA_COLUMNS:
        noise = np.random.normal(0, 0.1, size=len(prod_df))
        prod_df[col] = prod_df[col] + noise

    # Simulate some prediction errors
    prod_df["prediction"] = prod_df["target"].copy()
    error_mask = np.random.random(len(prod_df)) < 0.15  # 15% error rate
    prod_df.loc[error_mask, "prediction"] = 1 - prod_df.loc[error_mask, "prediction"]

    prod_df["user_feedback"] = prod_df["target"]

    return prod_df


def create_workspace() -> Workspace:
    """Create or get Evidently workspace."""
    ws = Workspace.create(WORKSPACE_PATH)
    return ws


def create_project(ws: Workspace) -> Project:
    """Create a new Evidently project."""
    project = ws.create_project(PROJECT_NAME)
    project.description = PROJECT_DESCRIPTION
    project.save()
    return project


def generate_data_drift_report(reference: pd.DataFrame, production: pd.DataFrame) -> Report:
    """Generate a data drift report."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=production,
        column_mapping=column_mapping
    )
    return report


def generate_classification_report(reference: pd.DataFrame, production: pd.DataFrame) -> Report:
    """Generate a classification performance report."""
    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=reference,
        current_data=production,
        column_mapping=column_mapping
    )
    return report


def generate_combined_report(reference: pd.DataFrame, production: pd.DataFrame) -> Report:
    """Generate a combined report with both drift and classification metrics."""
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])
    report.run(
        reference_data=reference,
        current_data=production,
        column_mapping=column_mapping
    )
    return report


def main():
    """Main function to set up Evidently project and generate reports."""
    print("=" * 60)
    print("Evidently Reporting Setup")
    print("=" * 60)

    # Load data
    print("\n1. Loading reference data...")
    ref_data = load_reference_data()
    print(f"   Reference data shape: {ref_data.shape}")

    print("\n2. Loading production data...")
    prod_data = load_production_data()
    print(f"   Production data shape: {prod_data.shape}")

    # Create workspace and project
    print("\n3. Creating Evidently workspace and project...")
    ws = create_workspace()
    project = create_project(ws)

    # Generate reports
    print("\n4. Generating Data Drift report...")
    drift_report = generate_data_drift_report(ref_data, prod_data)
    ws.add_report(project.id, drift_report)
    print("   Data Drift report added.")

    print("\n5. Generating Classification Performance report...")
    classification_report = generate_classification_report(ref_data, prod_data)
    ws.add_report(project.id, classification_report)
    print("   Classification report added.")

    print("\n6. Generating Combined report...")
    combined_report = generate_combined_report(ref_data, prod_data)
    ws.add_report(project.id, combined_report)
    print("   Combined report added.")

    project.save()

    print("\n" + "=" * 60)
    print("Evidently project setup complete!")
    print(f"Dashboard available at http://localhost:8082/")
    print("=" * 60)


if __name__ == "__main__":
    main()
