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

COUNTRY_MAP = {
    "Etats-Unis": "United States", 
    "Inde": "India", 
    "Canada": "Canada",
    "Royaume-Uni": "United Kingdom", 
    "Allemagne": "Germany",
    "Australie": "Australia", 
    "Afrique du Sud": "South Africa",
    "Japon": "Japan", 
    "Bresil": "Brazil", 
    "France": "France"
}

STRESS_MAP = {
    "Faible": "Low", 
    "Moyen": "Medium", 
    "Eleve": "High", 
    "Severe": "Severe"
}

YES_NO_MAP = {
    "Oui": "Yes", 
    "Non": "No"
}

PHYSICAL_ACTIVITY_MAP = {
    "Faible": "Low", 
    "Moderee": "Moderate", 
    "Elevee": "High"
}

TREATMENT_MAP = {
    "Therapie": "Therapy", 
    "Medicaments": "Medication", 
    "Les deux": "Both", 
    "Aucun": None
}

WORK_STATUS_MAP = {
    "Employe": "Employed", 
    "Sans emploi": "Unemployed", 
    "Etudiant": "Student", 
    "Retraite": "Retired"
}

GAD7_OPTIONS = {
    "Jamais": 0,
    "Plusieurs jours": 1,
    "Plus de la moitie des jours": 2,
    "Presque tous les jours": 3
}

GAD7_QUESTIONS = [
    "Se sentir nerveux(se), anxieux(se) ou a cran",
    "Ne pas etre capable d'arreter de s'inquieter ou de controler ses inquietudes",
    "S'inquieter de maniere excessive a propos de differentes choses",
    "Avoir du mal a se detendre",
    "Etre si agite(e) qu'il est difficile de rester assis(e)",
    "Devenir facilement contrarie(e) ou irritable",
    "Avoir peur que quelque chose de terrible puisse arriver"
]

PHQ9_OPTIONS = {
    "Jamais": 0,
    "Plusieurs jours": 1,
    "Plus de la moitie des jours": 2,
    "Presque tous les jours": 3
}

PHQ9_QUESTIONS = [
    "Peu d'interet ou de plaisir a faire les choses",
    "Se sentir triste, deprimé(e) ou sans espoir",
    "Avoir du mal a s'endormir ou a rester endormi(e), ou dormir trop",
    "Se sentir fatigué(e) ou n'avoir aucune energie",
    "Avoir un appétit diminué ou excessif",
    "Se sentir mal dans sa peau - ou que l'on est un échec ou que l'on a déçu sa famille",
    "Avoir du mal à se concentrer sur des choses comme lire le journal ou regarder la télévision",
    "Parler ou bouger si lentement que les autres pourraient l'avoir remarqué, ou être tellement agité(e) que vous bougez beaucoup plus que d'habitude",
    "Avoir des pensées que vous seriez mieux mort(e) ou de vous faire du mal d'une manière ou d'une autre"
]

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


# Main => Form
st.header("Informations du patient")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Genre", list(GENDER_MAP.keys()))
    country = st.selectbox("Pays", list(COUNTRY_MAP.keys()))
    stress_level = st.selectbox("Niveau de stress", list(STRESS_MAP.keys()))
    sleep_hours = st.number_input("Heures de sommeil (par nuit)", min_value=3.0, max_value=12.0, value=7.5, step=0.5)

with col2:
    physical_activity = st.selectbox("Activite physique", list(PHYSICAL_ACTIVITY_MAP.keys()))
    chronic_illness = st.selectbox("Maladie chronique", list(YES_NO_MAP.keys()))
    mental_health_history = st.selectbox("Antecedents de sante mentale", list(YES_NO_MAP.keys()))

with col3:
    treatment = st.selectbox("Traitement en cours", list(TREATMENT_MAP.keys()))
    days_of_treatment = st.number_input("Jours de traitement", min_value=0, max_value=365, value=0, step=1)
    work_status = st.selectbox("Situation professionnelle", list(WORK_STATUS_MAP.keys()))

# GAD-7
st.header("Questionnaire GAD-7")
st.markdown("Au cours des **2 dernieres semaines**, a quelle frequence avez-vous ete gene(e) par les problemes suivants ?")

gad7_scores = []
for i, question in enumerate(GAD7_QUESTIONS):
    score = st.selectbox(
        f"{i+1}. {question}",
        options=list(GAD7_OPTIONS.keys()),
        key=f"gad7_{i}"
    )
    gad7_scores.append(GAD7_OPTIONS[score])

anxiety_score = sum(gad7_scores)

# Interpretation du score
if anxiety_score <= 4:
    gad7_interpretation = "Anxiete minimale"
elif anxiety_score <= 9:
    gad7_interpretation = "Anxiete legere"
elif anxiety_score <= 14:
    gad7_interpretation = "Anxiete moderee"
else:
    gad7_interpretation = "Anxiete severe"

st.metric("Score GAD-7 total", f"{anxiety_score} / 21 -- {gad7_interpretation}")


# PHQ 9
st.header("Questionnaire PHQ-9")
st.markdown("Au cours des **2 dernieres semaines**, a quelle frequence avez-vous ete gene(e) par les problemes suivants ?")

phq9_scores = []
for i, question in enumerate(PHQ9_QUESTIONS):
    score = st.selectbox(
        f"{i+1}. {question}",
        options=list(PHQ9_OPTIONS.keys()),
        key=f"phq9_{i}"
    )
    phq9_scores.append(PHQ9_OPTIONS[score])

depression_score = sum(phq9_scores)

st.metric("Score PHQ-9 total", f"{depression_score} / 27")

# Optionnel : Email pour notification
user_email = st.text_input("Email utilisateur (pour notification par agent IA)", placeholder="utilisateur@exemple.com")

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
        "Country": COUNTRY_MAP[country],
        "Depression_Score": depression_score,
        "Anxiety_Score": anxiety_score,
        "Stress_Level": STRESS_MAP[stress_level],
        "Sleep_Hours": sleep_hours,
        "Physical_Activity": PHYSICAL_ACTIVITY_MAP[physical_activity],
        "Chronic_Illness": YES_NO_MAP[chronic_illness],
        "Mental_Health_History": YES_NO_MAP[mental_health_history],
        "Treatment": TREATMENT_MAP[treatment],
        "Days_of_Treatment": days_of_treatment,
        "Work_Status": WORK_STATUS_MAP[work_status]
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

        gad7_level = "minimal" if anxiety_score <= 4 else "leger" if anxiety_score <= 9 else "modere" if anxiety_score <= 14 else "severe"
        phq9_level = "minimal" if depression_score <= 4 else "leger" if depression_score <= 9 else "modere" if depression_score <= 14 else "moderement severe" if depression_score <= 19 else "severe"
        st.markdown("""
        **Facteurs cles influencant cette prediction :**
        - **Score GAD-7 (anxiete)** : {} / 21 -- niveau {}
        - **Score PHQ-9 (depression)** : {} / 27 -- niveau {}
        - **Niveau de stress** : {}
        - **Heures de sommeil** : {:.1f}h/nuit ({})
        - **Activite physique** : {}
        - **Maladie chronique** : {}
        - **Antecedents de sante mentale** : {}
        - **Traitement** : {} ({} jours)
        - **Situation professionnelle** : {}
        """.format(
            anxiety_score, gad7_level,
            depression_score, phq9_level,
            stress_level,
            sleep_hours,
            "en dessous des recommandations" if sleep_hours < 7 else "adequat",
            physical_activity,
            chronic_illness,
            mental_health_history,
            treatment, days_of_treatment,
            work_status
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