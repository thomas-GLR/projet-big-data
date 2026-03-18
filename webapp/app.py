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

# Questions DASS-42 (éléments sélectionnés)
DASS42_QUESTIONS = {
    "Q3A": "Je n'arrivais pas à ressentir aucun sentiment positif du tout",
    "Q10A": "J'avais l'impression que je n'avais rien à attendre de l'avenir",
    "Q13A": "Je me sentais triste et déprimé",
    "Q16A": "J'avais l'impression d'avoir perdu l'intérêt pour presque tout",
    "Q26A": "Je me sentais le cœur lourd et mélancolique",
    "Q34A": "J'avais l'impression d'être rather sans valeur",
    "Q37A": "Je ne voyais rien dans l'avenir auquel j'aurais pu m'accrocher",
    "Q38A": "J'avais l'impression que la vie était sans sens"
}

DASS42_OPTIONS = {
    "Ne s'appliquait pas du tout à moi": 0,
    "S'appliquait à moi dans une certaine mesure, ou à certains moments": 1, 
    "S'appliquait à moi dans une grande mesure, ou pendant une bonne partie du temps": 2,
    "S'appliquait à moi beaucoup, ou la plupart du temps": 3
}

# Mappages des champs démographiques avec labels
EDUCATION_OPTIONS = {
    "Moins que le secondaire": 1,
    "Secondaire": 2,
    "Diplôme universitaire": 3,
    "Diplôme d'études supérieures": 4
}

URBAN_OPTIONS = {
    "En campagne": 1,
    "Banlieue": 2,
    "Zone urbaine (ville, centre-ville)": 3
}

GENDER_OPTIONS = {
    "Homme": 1,
    "Femme": 2,
    "Autre": 3
}

HAND_OPTIONS = {
    "Droitier": 1,
    "Gaucher": 2,
    "Ambidextre": 3
}

RELIGION_OPTIONS = {
    "Agnostique": 1,
    "Athée": 2,
    "Bouddhiste": 3,
    "Chrétien (Catholique)": 4,
    "Chrétien (Mormon)": 5,
    "Chrétien (Protestant)": 6,
    "Chrétien (Autre)": 7,
    "Hindou": 8,
    "Juif": 9,
    "Musulman": 10,
    "Sikh": 11,
    "Autre": 12
}

ORIENTATION_OPTIONS = {
    "Hétérosexuel": 1,
    "Bisexuel": 2,
    "Homosexuel": 3,
    "Asexuel": 4,
    "Autre": 5
}

RACE_OPTIONS = {
    "Asiatique": 10,
    "Arabe": 20,
    "Noir": 30,
    "Australien Autochtone": 40,
    "Amérindien": 50,
    "Blanc": 60,
    "Autre": 70
}

VOTED_OPTIONS = {
    "Oui": 1,
    "Non": 2
}

MARRIED_OPTIONS = {
    "Jamais marié": 1,
    "Actuellement marié": 2,
    "Précédemment marié": 3
}

st.set_page_config(
    page_title="Prédiction Santé Mentale",
    layout="wide"
)

LABELS_ORDER = ["None", "Mild", "Moderate", "Severe", "Extremely severe"]
DIAGNOSIS_OPTIONS = {label: idx for idx, label in enumerate(LABELS_ORDER)}

# Session state initialization
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

page = st.sidebar.radio("Navigation", ["Prédiction", "Saisie Docteur"])

# Sidebar
with st.sidebar:
    st.header("Statut du Système")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"Statut API : {health['status']}")

        model_info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.metric("Total retours", model_info.get("total_feedbacks", 0))
        st.metric("Prochain réentraînement", model_info.get("next_retrain_at", "N/A"))
        st.metric("Type du modèle", model_info.get("model_type", "N/A"))
    except Exception as e:
        st.error(f"API non joignable : {e}")

    st.header("Historique des Prédictions")
    if st.session_state.predictions_history:
        hist_df = pd.DataFrame(st.session_state.predictions_history)
        # Distribution des prédictions
        fig_hist = px.histogram(
            hist_df, x="prediction_label",
            color="prediction_label",
            title="Distribution des Prédictions",
            color_discrete_map={"Yes": "#ff6b6b", "No": "#51cf66"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Score de risque au fil du temps
        fig_proba = px.line(
            hist_df, y="probability_yes",
            title="Score de Risque au Fil du Temps",
            labels={"probability_yes": "P(Oui)", "index": "Prédiction #"}
        )
        fig_proba.add_hline(y=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_proba, use_container_width=True)

        # Taux d'alerte
        alert_rate = len(hist_df[hist_df["prediction"] == 1]) / len(hist_df) * 100
        st.metric("Taux d'Alerte", f"{alert_rate:.1f}%")
    else:
        st.info("Aucune prédiction pour le moment.")

if page == "Saisie Docteur":
    st.title("Saisie Docteur - Validation D'Expert")
    st.write("Interface permettant aux docteurs de vérifier et de renseigner manuellement les résultats réels.")

    if st.session_state.predictions_history:
        for i, pred in enumerate(reversed(st.session_state.predictions_history)):
            with st.expander(f"Cas #{len(st.session_state.predictions_history)-i} - Prédiction IA : {pred['prediction_label']} ({pred.get('timestamp', 'N/A')[:16]})"):
                st.write(f"**Confiance du niveau prédit**: {pred.get('proba_label', 0):.2%}")
                if "input_data" in pred:
                    st.json(pred["input_data"])

                real_res = st.radio(
                    "Résultat réel du diagnostic :",
                    ["En attente"] + list(DIAGNOSIS_OPTIONS.keys()),
                    key=f"radio_{i}"
                )
                
                if st.button("Valider ce résultat", key=f"btn_val_{i}"):
                    if real_res == "En attente":
                        st.warning("Veuillez d'abord sélectionner le résultat réel.")
                    else:
                        feed_val = DIAGNOSIS_OPTIONS[real_res]
                        payload = {
                            "embedding": pred["embedding"],
                            "prediction": pred["prediction"],
                            "user_feedback": feed_val
                        }
                        try:
                            resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
                            if resp.status_code == 200:
                                st.success("Résultat manuel enregistré avec succès. Le modèle s'améliorera grâce à ce retour.")
                            else:
                                st.error("Erreur lors de l'enregistrement.")
                        except Exception as e:
                            st.error(f"API injoignable: {e}")
    else:
        st.info("Il n'y a pas encore de données ou de prédictions récentes à valider.")
        
    st.stop()


# Formulaire Principal
st.header("Informations Démographiques")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Quel âge avez-vous ?", min_value=17, max_value=90, value=30)
    
    education_label = st.selectbox(
        "Quel est votre niveau d'études ?",
        options=list(EDUCATION_OPTIONS.keys())
    )
    education = EDUCATION_OPTIONS[education_label]
    
    gender_label = st.selectbox(
        "Quel est votre genre ?",
        options=list(GENDER_OPTIONS.keys())
    )
    gender = GENDER_OPTIONS[gender_label]
    
    hand_label = st.selectbox(
        "Quelle main utilisez-vous pour écrire ?",
        options=list(HAND_OPTIONS.keys())
    )
    hand = HAND_OPTIONS[hand_label]
    
with col2:
    urban_label = st.selectbox(
        "Dans quel type de zone avez-vous grandi ?",
        options=list(URBAN_OPTIONS.keys())
    )
    urban = URBAN_OPTIONS[urban_label]
    
    religion_label = st.selectbox(
        "Quelle est votre religion ?",
        options=list(RELIGION_OPTIONS.keys())
    )
    religion = RELIGION_OPTIONS[religion_label]
    
    orientation_label = st.selectbox(
        "Quelle est votre orientation sexuelle ?",
        options=list(ORIENTATION_OPTIONS.keys())
    )
    orientation = ORIENTATION_OPTIONS[orientation_label]
    
    race_label = st.selectbox(
        "Quelle est votre origine ethnique ?",
        options=list(RACE_OPTIONS.keys())
    )
    race = RACE_OPTIONS[race_label]

col3, col4, col5 = st.columns(3)

with col3:
    voted_label = st.selectbox(
        "Avez-vous voté aux élections nationales l'année dernière ?",
        options=list(VOTED_OPTIONS.keys())
    )
    voted_int = VOTED_OPTIONS[voted_label]

with col4:
    familysize = st.number_input("En vous incluant, combien d'enfants votre mère a-t-elle eu ?", min_value=1, max_value=20, value=2, step=1)

with col5:
    married_label = st.selectbox(
        "Quel est votre statut matrimonial ?",
        options=list(MARRIED_OPTIONS.keys())
    )
    married = MARRIED_OPTIONS[married_label]

# Questionnaire DASS-42
st.header("Questionnaire DASS-42")
st.markdown("Veuillez évaluer votre accord avec chaque affirmation selon l'échelle suivante")

dass42_scores = []
for question_id, question_text in DASS42_QUESTIONS.items():
    score = st.selectbox(
        f"{question_id}: {question_text}",
        options=list(DASS42_OPTIONS.keys()),
        key=f"dass42_{question_id}"
    )
    dass42_scores.append(DASS42_OPTIONS[score])

dass42_total = sum(dass42_scores)
st.metric("Score DASS-42 Total", f"{dass42_total} / 24")

# Optionnel : Email pour notification
user_email = st.text_input("Email (pour notification par agent IA)", placeholder="utilisateur@exemple.com")

# Bouton de Prédiction
col_pred, col_notify = st.columns(2)

with col_pred:
    predict_btn = st.button("Prédire", type="primary", use_container_width=True)

with col_notify:
    notify_btn = st.button("Notifier l'utilisateur (Agent IA)", use_container_width=True,
                           disabled=st.session_state.last_prediction is None)

# Handle Prediction 
if predict_btn:
    input_data = {
        "Q3A": dass42_scores[0],
        "Q10A": dass42_scores[1],
        "Q13A": dass42_scores[2],
        "Q16A": dass42_scores[3],
        "Q26A": dass42_scores[4],
        "Q34A": dass42_scores[5],
        "Q37A": dass42_scores[6],
        "Q38A": dass42_scores[7],
        "age": age,
        "voted": voted_int,
        "familysize": familysize,
        "education": education,
        "urban": urban,
        "gender": gender,
        "hand": hand,
        "religion": religion,
        "orientation": orientation,
        "race": race,
        "married": married
    }

    try:
        with st.spinner("Appel de l'API de prédiction..."):
            response = requests.post(f"{API_URL}/predict", json=input_data, timeout=10)
            if response.status_code != 200:
                st.error(f"Erreur API ({response.status_code})")
                st.write("Détails:", response.text)
                st.stop()
            result = response.json()

        prediction_proba = result.get("prediction_proba", [])
        prediction_label = result.get("prediction_label", "Inconnu")
        if prediction_proba:
            pred_idx = int(max(range(len(prediction_proba)), key=lambda i: prediction_proba[i]))
            predicted_score = float(prediction_proba[pred_idx])
            none_score = float(prediction_proba[0]) if len(prediction_proba) > 0 else 0.0
            risk_score = max(0.0, min(1.0, 1.0 - none_score))
        else:
            pred_idx = 0
            predicted_score = float(result.get("proba_label", 0.0))
            risk_score = 0.0

        normalized_result = {
            **result,
            "prediction": pred_idx,
            "probability_yes": risk_score,
            "probability_no": 1.0 - risk_score
        }

        st.session_state.last_prediction = {
            **normalized_result,
            "input_data": input_data,
            "timestamp": datetime.now().isoformat(),
            "user_email": user_email
        }

        # Ajout à l'historique
        st.session_state.predictions_history.append({
            **normalized_result,
            "timestamp": datetime.now().isoformat()
        })

        # Affichage du résultat
        st.header("Résultat de la Prédiction")

        if prediction_label in ["Severe", "Extremely severe"]:
            st.error(f"Niveau détecté: {prediction_label}")
        elif prediction_label == "Moderate":
            st.warning(f"Niveau détecté: {prediction_label}")
        else:
            st.success(f"Niveau détecté: {prediction_label}")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Confiance du niveau prédit", f"{predicted_score:.2%}")
        with res_col2:
            st.metric("Score de risque global", f"{risk_score:.2%}")
        with res_col3:
            st.metric("Dimension embedding", len(result.get("embedding", [])))

        proba_labels = LABELS_ORDER[:len(prediction_proba)] if prediction_proba else [prediction_label]
        proba_values = prediction_proba if prediction_proba else [predicted_score]
        proba_df = pd.DataFrame({"Niveau": proba_labels, "Probabilité": proba_values})

        fig_proba_detail = px.bar(
            proba_df,
            x="Niveau",
            y="Probabilité",
            text=proba_df["Probabilité"].map(lambda x: f"{x:.1%}"),
            color="Probabilité",
            color_continuous_scale="Blues",
            title="Répartition des probabilités par niveau"
        )
        fig_proba_detail.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
        fig_proba_detail.update_traces(textposition="outside")
        st.plotly_chart(fig_proba_detail, use_container_width=True)

        with st.expander("Voir les détails techniques"):
            st.write("Index de classe prédit:", pred_idx)
            st.write("Probabilités brutes:", prediction_proba)
            st.write("Aperçu embedding (10 premières valeurs):", result.get("embedding", [])[:10])

    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter à l'API de prédiction. Vérifiez que le conteneur serving est en cours d'exécution.")
    except Exception as e:
        st.error(f"Erreur : {e}")

# Gestion de la Notification (Agent IA)
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
                st.success("Notification envoyée avec succès via l'agent IA !")
                st.info("L'utilisateur recevra un email avec les détails de la prédiction et un lien pour donner son avis.")
            else:
                st.warning(f"L'agent a répondu avec le statut {response.status_code} : {response.text}")

        except requests.exceptions.ConnectionError:
            st.warning("Agent n8n non joignable. Vérifiez que le conteneur n8n est en cours d'exécution.")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi de la notification : {e}")