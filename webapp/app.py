import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Configuration
API_URL = "http://serving-api:8080"
N8N_WEBHOOK_URL = "http://n8n:5678/webhook/notify-user"

# Traductions (francais -> anglais pour l'API)
GENDER_MAP = {
    "Homme": "Male", 
    "Femme": "Female"
    }

OCCUPATION_MAP = {
    "Informatique": "IT", 
    "Ingenierie": "Engineering", 
    "Sante": "Healthcare",
    "Education": "Education", 
    "Finance": "Finance", 
    "Ventes": "Sales", 
    "Autre": "Other"
}

COUNTRY_MAP = {
    "Etats-Unis": "USA", 
    "Inde": "India", 
    "Canada": "Canada",
    "Royaume-Uni": "UK", 
    "Allemagne": "Germany", 
    "Australie": "Australia", 
    "Autre": "Other"
}

SEVERITY_MAP = {
    "Aucune": "None", 
    "Faible": "Low", 
    "Moyenne": "Medium", 
    "Elevee": "High"
}

YES_NO_MAP = {
    "Oui": "Yes", 
    "Non": "No"
}

STRESS_MAP = {
    "Faible": "Low", 
    "Moyen": "Medium", 
    "Eleve": "High"
}

st.set_page_config(
    page_title="Prediction Sante Mentale",
    layout="wide"
)

# Session state initialization
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# Sidebar
with st.sidebar:
    st.header("Statut du systeme")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"Statut API : {health['status']}")

        model_info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.metric("Total retours", model_info.get("total_feedbacks", 0))
        st.metric("Prochain re-entrainement", model_info.get("next_retrain_at", "N/A"))
        st.metric("Type de modele", model_info.get("model_type", "N/A"))
    except Exception as e:
        st.error(f"API non joignable : {e}")

    st.header("Historique des predictions")
    if st.session_state.predictions_history:
        hist_df = pd.DataFrame(st.session_state.predictions_history)
        # Distribution des predictions
        fig_hist = px.histogram(
            hist_df, x="prediction_label",
            color="prediction_label",
            title="Distribution des predictions",
            color_discrete_map={"Yes": "#ff6b6b", "No": "#51cf66"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Scores de probabilite dans le temps
        fig_proba = px.line(
            hist_df, y="probability_yes",
            title="Score de risque dans le temps",
            labels={"probability_yes": "P(Oui)", "index": "Prediction n"}
        )
        fig_proba.add_hline(y=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_proba, use_container_width=True)

        # Taux d'alerte
        alert_rate = len(hist_df[hist_df["prediction"] == 1]) / len(hist_df) * 100
        st.metric("Taux d'alerte", f"{alert_rate:.1f}%")
    else:
        st.info("Aucune prediction pour le moment.")


# Main => Formulaire de saisie
st.header("Informations du patient")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Genre", list(GENDER_MAP.keys()))
    occupation = st.selectbox("Profession", list(OCCUPATION_MAP.keys()))

with col2:
    country = st.selectbox("Pays", list(COUNTRY_MAP.keys()))
    severity = st.selectbox("Severite (symptomes actuels)", list(SEVERITY_MAP.keys()))
    consultation_history = st.selectbox("Consultation precedente", list(YES_NO_MAP.keys()))

with col3:
    stress_level = st.selectbox("Niveau de stress", list(STRESS_MAP.keys()))
    sleep_hours = st.number_input("Heures de sommeil (par nuit)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    work_hours = st.number_input("Heures de travail (par semaine)", min_value=0.0, max_value=120.0, value=40.0, step=1.0)

physical_activity = st.slider("Activite physique (heures/semaine)", min_value=0, max_value=20, value=3)

# Optionnel : Email pour notification
user_email = st.text_input("Email utilisateur (pour notification par agent IA)", placeholder="utilisateur@exemple.com")

st.markdown("---")

# Prediction Button
col_pred, col_notify = st.columns(2)

with col_pred:
    predict_btn = st.button("Predire", type="primary", use_container_width=True)

with col_notify:
    notify_btn = st.button("Notifier l'utilisateur (Agent IA)", use_container_width=True,
                           disabled=st.session_state.last_prediction is None)

# Handle Prediction 
if predict_btn:
    input_data = {
        "Age": age,
        "Gender": GENDER_MAP[gender],
        "Occupation": OCCUPATION_MAP[occupation],
        "Country": COUNTRY_MAP[country],
        "Severity": SEVERITY_MAP[severity],
        "Consultation_History": YES_NO_MAP[consultation_history],
        "Stress_Level": STRESS_MAP[stress_level],
        "Sleep_Hours": sleep_hours,
        "Work_Hours": work_hours,
        "Physical_Activity_Hours": physical_activity
    }

    try:
        with st.spinner("Appel de l'API de prediction..."):
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

        # Affichage du resultat
        st.markdown("---")
        st.header("Resultat de la prediction")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if result["prediction"] == 1:
                st.error(f"Condition de sante mentale : **{result['prediction_label']}**")
            else:
                st.success(f"Condition de sante mentale : **{result['prediction_label']}**")

        with res_col2:
            st.metric("Probabilite (Oui)", f"{result['probability_yes']:.2%}")

        with res_col3:
            st.metric("Probabilite (Non)", f"{result['probability_no']:.2%}")

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability_yes"] * 100,
            title={"text": "Score de risque (%)"},
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

        # Interpretation des facteurs
        st.subheader("Interpretation des facteurs")
        st.markdown("""
        **Facteurs cles influencant cette prediction :**
        - **Niveau de stress** : un stress {} peut augmenter significativement le risque
        - **Heures de sommeil** : {:.1f}h/nuit ({})
        - **Heures de travail** : {:.1f}h/semaine ({})
        - **Historique de consultation** : {} consultation prealable
        - **Severite des symptomes actuels** : {}
        """.format(
            stress_level,
            sleep_hours,
            "en dessous des recommandations" if sleep_hours < 7 else "adequat",
            work_hours,
            "au-dessus de la moyenne" if work_hours > 45 else "dans la normale",
            "A eu une" if consultation_history == "Oui" else "Aucune",
            severity
        ))

    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter a l'API de prediction. Verifiez que le conteneur serving est en cours d'execution.")
    except Exception as e:
        st.error(f"Erreur : {e}")

# Handle Notification (AI Agent)
if notify_btn and st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    if not pred.get("user_email"):
        st.warning("Veuillez fournir une adresse email pour notifier l'utilisateur.")
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

            with st.spinner("Envoi de la notification via l'agent IA (n8n)..."):
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)

            if response.status_code == 200:
                st.success("Notification envoyee avec succes via l'agent IA !")
                st.info("L'utilisateur recevra un email avec les details de la prediction et un lien pour donner son retour.")
            else:
                st.warning(f"L'agent a repondu avec le statut {response.status_code} : {response.text}")

        except requests.exceptions.ConnectionError:
            st.warning("Agent n8n non joignable. Verifiez que le conteneur n8n est en cours d'execution.")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi de la notification : {e}")