# 🧠 Mental Health Prediction - MLOps Project

## Projet BI/DATA — Mise en production et déploiement continu
**Année universitaire 2025-2026 | 5A APP | Enseignant: Haytham Elghazel**

---

## 📋 Description

Pipeline MLOps complet pour la prédiction de conditions de santé mentale, incluant :
- **Modèle ML** : RandomForest sur données embarquées (PCA)
- **API de Serving** : FastAPI (port 8080)
- **Interface Web** : Streamlit (port 8081)
- **Reporting/Monitoring** : Evidently AI (port 8082)
- **Agent IA** : n8n + LLM (port 5678)
- **Boucle Human-in-the-Loop** : feedback par email -> réentraînement automatique

---

## 🏗️ Architecture du Projet

```
PROJET/
├── docker-compose.yml          # Orchestration globale de tous les services
├── mental_health_dataset.csv   # Dataset brut
├── README.md
│
├── scripts/                    # Code factorisé et notebooks
│   └── model_utils.py          # Fonctions de preprocessing, training, etc.
│
├── data/                       # Données transformées
│   ├── ref_data.csv            # Données de référence (PCA embedded)
│   └── prod_data.csv           # Données de production (feedback utilisateurs)
│
├── artifacts/                  # Modèles et artefacts sauvegardés
│   ├── model.pkl               # Modèle RandomForest
│   ├── scaler.pkl              # StandardScaler
│   ├── pca.pkl                 # Modèle PCA
│   └── label_encoders.pkl      # Encodeurs de labels
│
├── serving/                    # API de serving FastAPI
│   ├── api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── docker-compose.yml
│
├── webapp/                     # Interface web Streamlit
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── docker-compose.yml
│
├── reporting/                  # Reporting Evidently
│   ├── project.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── docker-compose.yml
│
└── n8n/                        # Workflow n8n (AI Agent)
    └── workflow.json           # Workflow à importer dans n8n
```

---

## 🚀 Démarrage Rapide

### Prérequis
- [Docker Desktop](https://docs.docker.com/desktop/) installé
- Dataset `mental_health_dataset.csv` dans le dossier `PROJET/`

### 1. Préparer les données et entraîner le modèle

Exécuter le notebook `projet.ipynb` pour :
- Créer `data/ref_data.csv` (embedding PCA)
- Entraîner le modèle et sauvegarder les artifacts

### 2. Lancer tous les services avec Docker Compose

```bash
# Depuis le dossier PROJET/
docker compose up --build
```

Ou service par service :

```bash
# 1. API de serving
docker compose -f serving/docker-compose.yml up --build

# 2. Interface web (dans un autre terminal)
docker compose -f webapp/docker-compose.yml up --build

# 3. Reporting Evidently
docker compose -f reporting/docker-compose.yml up --build
```

### 3. Accéder aux services

| Service | URL | Description |
|---------|-----|-------------|
| API FastAPI | http://localhost:8080/docs | Documentation Swagger de l'API |
| Streamlit | http://localhost:8081 | Interface web utilisateur |
| Evidently | http://localhost:8082 | Dashboard de monitoring |
| n8n | http://localhost:5678 | Interface n8n (Agent IA) |

---

## 📊 Pipeline de Données

```
Données brutes (CSV)
    ↓ LabelEncoding (catégorielles)
    ↓ StandardScaler (normalisation)
    ↓ PCA (5 composantes = embedding)
    ↓ RandomForest (classification)
    → Prédiction (Yes/No + probabilités)
```

### Colonnes du fichier ref_data.csv / prod_data.csv
| Colonne | Description |
|---------|-------------|
| pca_0 ... pca_4 | Composantes PCA (embedding) |
| target | Label réel (0=No, 1=Yes) |
| prediction | Prédiction du modèle |
| user_feedback | Retour utilisateur (Human-in-the-Loop) |

---

## 🔄 Boucle de Réentraînement

1. L'utilisateur saisit des données via **Streamlit**
2. L'API **FastAPI** retourne une prédiction
3. L'utilisateur clique sur "Notifier" → **n8n** envoie un email via **LLM**
4. L'utilisateur confirme/corrige via le lien dans l'email
5. Le feedback est enregistré dans `prod_data.csv`
6. Tous les **10 feedbacks**, le modèle est automatiquement réentraîné

---

## 🤖 Agent IA (n8n + LLM)

1. Importer le workflow `n8n/workflow.json` dans n8n
2. Configurer les credentials :
   - **OpenAI API Key** (pour le LLM)
   - **SMTP** (pour l'envoi d'emails)
3. Activer le workflow

Le workflow n8n :
- Reçoit la prédiction via webhook
- Génère un email personnalisé via GPT
- Envoie l'email à l'utilisateur
- L'utilisateur clique sur un lien pour valider/corriger
- Le feedback est renvoyé à l'API de serving

---

## 📈 Monitoring avec Evidently

Le dashboard Evidently génère des rapports sur :
- **Data Drift** : détection de dérive entre données de référence et production
- **Classification Performance** : F1 Score, Balanced Accuracy, Precision, Recall

---

## 🛠️ API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Status de l'API |
| GET | `/health` | Health check |
| POST | `/predict` | Prédiction sur une donnée |
| POST | `/feedback` | Soumission de feedback utilisateur |
| GET | `/model-info` | Informations sur le modèle |
| POST | `/webhook/feedback` | Webhook pour n8n |
| GET | `/feedback-form` | Formulaire HTML de feedback |
