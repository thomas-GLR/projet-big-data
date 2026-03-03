"""
Streamlit Web Application for Mental Health Prediction.
Provides a user interface to interact with the prediction API,
visualize results, and trigger AI agent notifications.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# --- Configuration ---
API_URL = "http://serving-api:8080"
N8N_WEBHOOK_URL = "http://n8n:5678/webhook/notify-user"

st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health Condition Prediction")
st.markdown("---")

# --- Session state initialization ---
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# --- Sidebar: API Status ---
with st.sidebar:
    st.header("📊 System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API Status: {health['status']}")

        model_info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.metric("Total Feedbacks", model_info.get("total_feedbacks", 0))
        st.metric("Next Retrain At", model_info.get("next_retrain_at", "N/A"))
        st.metric("Model Type", model_info.get("model_type", "N/A"))
    except Exception as e:
        st.error(f"API not reachable: {e}")

    st.markdown("---")
    st.header("📈 Prediction History")
    if st.session_state.predictions_history:
        hist_df = pd.DataFrame(st.session_state.predictions_history)
        # Distribution of predictions
        fig_hist = px.histogram(
            hist_df, x="prediction_label",
            color="prediction_label",
            title="Prediction Distribution",
            color_discrete_map={"Yes": "#ff6b6b", "No": "#51cf66"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Probability scores over time
        fig_proba = px.line(
            hist_df, y="probability_yes",
            title="Risk Score Over Time",
            labels={"probability_yes": "P(Yes)", "index": "Prediction #"}
        )
        fig_proba.add_hline(y=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_proba, use_container_width=True)

        # Alert rate
        alert_rate = len(hist_df[hist_df["prediction"] == 1]) / len(hist_df) * 100
        st.metric("Alert Rate", f"{alert_rate:.1f}%")
    else:
        st.info("No predictions yet.")


# --- Main: Input Form ---
st.header("📝 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
    occupation = st.selectbox("Occupation", ["IT", "Engineering", "Healthcare", "Education", "Finance", "Sales", "Other"])

with col2:
    country = st.selectbox("Country", ["USA", "India", "Canada", "UK", "Germany", "Australia", "Other"])
    severity = st.selectbox("Severity (current symptoms)", ["None", "Low", "Medium", "High"])
    consultation_history = st.selectbox("Previous Consultation", ["Yes", "No"])

with col3:
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    work_hours = st.number_input("Work Hours (per week)", min_value=0.0, max_value=120.0, value=40.0, step=1.0)

physical_activity = st.slider("Physical Activity (hours/week)", min_value=0, max_value=20, value=3)

# Optional: Email for notification
user_email = st.text_input("📧 User Email (for AI agent notification)", placeholder="user@example.com")

st.markdown("---")

# --- Prediction Button ---
col_pred, col_notify = st.columns(2)

with col_pred:
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

with col_notify:
    notify_btn = st.button("📨 Notify User (AI Agent)", use_container_width=True,
                           disabled=st.session_state.last_prediction is None)

# --- Handle Prediction ---
if predict_btn:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Country": country,
        "Severity": severity,
        "Consultation_History": consultation_history,
        "Stress_Level": stress_level,
        "Sleep_Hours": sleep_hours,
        "Work_Hours": work_hours,
        "Physical_Activity_Hours": physical_activity
    }

    try:
        with st.spinner("Calling prediction API..."):
            response = requests.post(f"{API_URL}/predict", json=input_data, timeout=10)
            result = response.json()

        st.session_state.last_prediction = {
            **result,
            "input_data": input_data,
            "timestamp": datetime.now().isoformat(),
            "user_email": user_email
        }

        # Add to history
        st.session_state.predictions_history.append(result)

        # Display result
        st.markdown("---")
        st.header("🎯 Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if result["prediction"] == 1:
                st.error(f"⚠️ Mental Health Condition: **{result['prediction_label']}**")
            else:
                st.success(f"✅ Mental Health Condition: **{result['prediction_label']}**")

        with res_col2:
            st.metric("Probability (Yes)", f"{result['probability_yes']:.2%}")

        with res_col3:
            st.metric("Probability (No)", f"{result['probability_no']:.2%}")

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability_yes"] * 100,
            title={"text": "Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if result["probability_yes"] > 0.5 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance interpretation
        st.subheader("📊 Feature Interpretation")
        st.markdown("""
        **Key factors influencing this prediction:**
        - **Stress Level**: {} stress can significantly increase risk
        - **Sleep Hours**: {:.1f}h/night ({})
        - **Work Hours**: {:.1f}h/week ({})
        - **Consultation History**: {} prior consultation
        - **Severity of Current Symptoms**: {}
        """.format(
            stress_level,
            sleep_hours,
            "below recommended" if sleep_hours < 7 else "adequate",
            work_hours,
            "above average" if work_hours > 45 else "normal range",
            "Has" if consultation_history == "Yes" else "No",
            severity
        ))

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the prediction API. Make sure the serving container is running.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Handle Notification (AI Agent) ---
if notify_btn and st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    if not pred.get("user_email"):
        st.warning("⚠️ Please provide an email address to notify the user.")
    else:
        try:
            payload = {
                "email": pred["user_email"],
                "prediction": pred["prediction"],
                "prediction_label": pred["prediction_label"],
                "probability_yes": pred["probability_yes"],
                "probability_no": pred["probability_no"],
                "embedding": pred["embedding"],
                "input_data": pred["input_data"],
                "timestamp": pred["timestamp"]
            }

            with st.spinner("Sending notification via AI Agent (n8n)..."):
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)

            if response.status_code == 200:
                st.success("✅ Notification sent successfully via AI Agent!")
                st.info("The user will receive an email with the prediction details and a link to provide feedback.")
            else:
                st.warning(f"⚠️ Agent responded with status {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.warning("⚠️ n8n agent not reachable. Make sure the n8n container is running.")
        except Exception as e:
            st.error(f"❌ Error sending notification: {e}")


# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "Mental Health Prediction MLOps Project — 2025-2026"
    "</div>",
    unsafe_allow_html=True
)
