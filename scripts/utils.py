import pandas as pd
import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pickle
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_validate
from sklearn.base import is_classifier, is_regressor

# --- Constants ---
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
DATASET_PATH = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "Global_Mental_Health_Dataset_2025.csv"))

PATIENT_ID = "Patient_ID"
AGE = "Age"
GENDER = "Gender"
COUNTRY = "Country"
ANXIETY_SCORE = "Anxiety_score"
STRESS_LEVEL = "Stress_Level"
SLEEP_HOURS = "Sleep_Hours"
PHYSICAL_ACTIVITY = "Physical_Activity"
CHRONIC_ILLNESS = "Chronic_Illness"
MENTAL_HEALTH_HISTORY = "Mental_Health_History"
TREATMENT = "Treatment"
DAYS_OF_TREATMENT = "Days_of_Treatment"
DEPRESSION_SCORE = "Depression_Score"
OUTCOME = "Outcome"
WORK_STATUS = "Work_Status"

USELESS_COLUMNS = [PATIENT_ID]

FEATURE_COLS = [AGE, GENDER, COUNTRY, ANXIETY_SCORE,
                STRESS_LEVEL, SLEEP_HOURS, PHYSICAL_ACTIVITY,
                CHRONIC_ILLNESS, MENTAL_HEALTH_HISTORY, TREATMENT,
                DAYS_OF_TREATMENT, DEPRESSION_SCORE, WORK_STATUS]

CATEGORICAL_COLS = [GENDER, COUNTRY, STRESS_LEVEL, PHYSICAL_ACTIVITY,
                    CHRONIC_ILLNESS, MENTAL_HEALTH_HISTORY, TREATMENT, WORK_STATUS]

BINARY_COLS = [GENDER, CHRONIC_ILLNESS, MENTAL_HEALTH_HISTORY, WORK_STATUS]

COLUMNS_TO_ENCODE = [COUNTRY, STRESS_LEVEL, PHYSICAL_ACTIVITY, TREATMENT]

TARGET_COL = OUTCOME


def load_raw_data_without_useless_columns(filepath: str = None) -> pd.DataFrame:
    """Load the raw dataset without useless columns."""
    if filepath is None:
        filepath = DATASET_PATH
    df = pd.read_csv(filepath, sep=",")
    df = df.drop(columns=USELESS_COLUMNS)

    return df


def target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target variable 'Outcome' in ordinal"""
    labels = ['Poor', 'Fair', 'Good', 'Excellent']

    order_map = {label: i for i, label in enumerate(labels)}
    # {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    df[TARGET_COL] = df[TARGET_COL].map(order_map)
    return df


def encode_categorical_columns(df: pd.DataFrame, target_encoder_columns=None, binary_columns=None) -> pd.DataFrame:
    """
    Encode categorical columns. For binary columns replace by 1 and 0. For columns with multiple values,
    use pd.get_dummies() with drop_first=True to avoid multicollinearity issues.
    Use the BINARY_COLS_MAPPER to map binary columns to 1 and 0,
    and DUMMIES_ENCODER_COLS for the rest of the categorical columns with multiple values.

    :param df: DataFrame with categorical columns to encode
    :param target_encoder_columns: List of columns to encode with TargetEncoder
    :param binary_columns: List of binary columns to encode with LabelEncoder
    :return: DataFrame with encoded categorical columns
    """
    if binary_columns is None:
        binary_columns = BINARY_COLS
    if target_encoder_columns is None:
        target_encoder_columns = COLUMNS_TO_ENCODE

    label_encoder = LabelEncoder()

    for col in binary_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Use drop_first=True in pd.get_dummies() before PCA (ACP) and ML classifiers to avoid multicollinearity issues.
    # Full dummies (drop_first=False) create redundant columns where one is predictable from others, causing perfect multicollinearity.
    # PCA amplifies this by producing near-zero variance components, and linear classifiers (e.g., logistic regression) suffer unstable coefficients.
    # Tree-based classifiers (e.g., Random Forest) tolerate it but benefit from fewer features.
    # df = pd.get_dummies(df, columns=DUMMIES_ENCODER_COLS, drop_first=True)

    target_encoder = TargetEncoder(target_type="multiclass", smooth="auto")

    Y = get_target(df)
    unique_labels = sorted(np.unique(Y))  # ['Excellent', 'Fair', 'Good', 'Poor']

    encoded = target_encoder.fit_transform(df[target_encoder_columns], Y)

    new_cols = [
        f"{col}_{label}"
        for col in target_encoder_columns
        for label in unique_labels
    ]

    encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
    df = df.drop(columns=target_encoder_columns)
    df = pd.concat([df, encoded_df], axis=1)

    return df


def get_target_classes(df: pd.DataFrame) -> np.ndarray:
    """Get the unique classes of the target variable."""
    return np.unique(df[TARGET_COL].values, sorted=True)


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract the target variable as a numpy array."""
    return df[TARGET_COL].values


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Divide the DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[TARGET_COL]).values
    # y = target_encoding(df).values
    y = get_target(df)
    features_label = df.drop(columns=[TARGET_COL]).columns.tolist()

    return X, y, features_label

def build_embedding(X: np.ndarray, n_pca_components: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    """
    Apply StandardScaler + PCA embedding.
    Returns embedded data, scaler, and pca objects.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_pca_components)
    X_embedded = pca.fit_transform(X_scaled)

    return X_embedded, scaler, pca


def save_artifact(obj, filename: str):
    """Save a Python object as a pickle file in the artifacts directory."""
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Artifact saved: {filepath}")


def create_ref_data(raw_filepath: str, output_filepath: str, n_pca_components: int):
    """
    Transform the raw dataset into ref_data.csv with PCA-embedded vectors.
    Also saves the label_encoders, scaler, and PCA model.
    """
    df = load_raw_data_without_useless_columns(raw_filepath)

    # Preprocess features
    df_encoded = encode_categorical_columns(df)
    x, y, label_encoders = split_features_target(df_encoded)

    # Build embedding
    x_embedded, scaler, pca = build_embedding(x, n_pca_components)

    # Create ref_data DataFrame
    pca_columns = [f"pca_{i}" for i in range(n_pca_components)]
    ref_df = pd.DataFrame(x_embedded, columns=pca_columns)
    ref_df["target"] = y

    # Save ref_data.csv
    ref_df.to_csv(output_filepath, index=False)
    print(f"ref_data.csv saved to {output_filepath} with shape {ref_df.shape}")

    # Save preprocessing artifacts
    save_artifact(label_encoders, "label_encoders.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(pca, "pca.pkl")

    return ref_df, label_encoders, scaler, pca


def normalisation_donnes(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def pca(X: np.ndarray, Y: np.ndarray, columns_label: list[str], n_components: float) -> PCA:
    # Labels of row so the target
    rows_label = Y.tolist()

    # Centrer et réduire les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_scaled)

    print("Explication des ratio de la variance:\n{}".format(pca.explained_variance_ratio_))

    variances = pca.explained_variance_
    print("Voici les variances des {} variables qui ont plus de {}% de différence :\n{}".format(len(variances), n_components * 100,
                                                                                                variances))

    pourcentages = pca.explained_variance_ratio_

    print("Voici les pourcentages des {} variables qui ont plus de {}% de différence :\n{}".format(len(pourcentages), n_components * 100,
                                                                                                   pourcentages))

    seuil = 1 / np.sqrt(len(columns_label)) - 0.03

    print("Seuil : {}".format(seuil))

    print("Valeur du premier axe : {}".format(pca.components_[0, :]))

    print("Graphique du premier axe : ")

    plt.figure(figsize=(40, 20))
    plt.bar(columns_label, pca.components_[0, :], color='skyblue')
    if any(x > 0 for x in pca.components_[0, :]):
        plt.axhline(seuil, color='red', linestyle='--', linewidth=2)
    if any(x < 0 for x in pca.components_[0, :]):
        plt.axhline(-seuil, color='red', linestyle='--', linewidth=2)
    plt.title("Variable y1")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Valeur du deuxieme axe : {}".format(pca.components_[1, :]))

    print("Graphique du deuxième axe : ")

    plt.figure(figsize=(40, 20))
    plt.bar(columns_label, pca.components_[1, :], color='skyblue')
    if any(x > 0 for x in pca.components_[1, :]):
        plt.axhline(seuil, color='red', linestyle='--', linewidth=2)
    if any(x < 0 for x in pca.components_[1, :]):
        plt.axhline(-seuil, color='red', linestyle='--', linewidth=2)
    plt.title("Variable y2")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    color_map = {"Excellent": 'red', "Fair": 'blue', "Good": 'green', "Poor": 'orange'}

    plt.figure(figsize=(40, 20))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[color_map[label] for label in rows_label])


    # Ajouter les labels sur chaque point
    for i, label in enumerate(rows_label):
        plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), fontsize=9, alpha=0.7)

    plt.xlabel('y1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 2)))
    plt.ylabel('y2 ({}%)'.format(round(pca.explained_variance_ratio_[1], 2)))
    plt.title('Projection PCA')
    plt.grid(True)
    plt.show()


def run_classifiers_cv(
        clfs_par_nom,
        X,
        Y
):
    """
    Effectue une cross-validation sur les classifieurs passés en paramètre et compare dans un tableau
    les moyennes et écrat type de :
      – l'accuracy
      – l'AUC
      – le temps d'exécution d'un fold
      – le score final (accuracy + précision) / 2

    :param clfs_par_nom: le dictionnaire des classifieurs avec en clé le nom du modèle et en valeur l'algorithme
    :param X: les données
    :param Y: les targets à prédire des données
    :return: le nom du meilleur modèle (modèle avec score final max) et son score final.
    """
    cv_scores_par_nom_model = {}

    result_from_all_model = []

    nombre_fold = 10

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": make_scorer(
            roc_auc_score,
            multi_class="ovr",
            average="macro",
            needs_proba=True
        ),
        "precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0
        ),
    }

    kf = KFold(n_splits=nombre_fold, shuffle=True, random_state=0)
    for nom_model, clf in clfs_par_nom.items():

        resultats = cross_validate(clf, X, Y, cv=kf, scoring=scoring)

        accuracy = resultats["test_accuracy"]
        auc = resultats["test_roc_auc"]
        precision = resultats["test_precision"]
        recall = resultats["test_recall"]

        score_time = resultats["score_time"]

        cv_scores_par_nom_model[nom_model] = np.mean(accuracy)

        result_from_all_model.append([
            accuracy.mean(),
            accuracy.std(),
            auc.mean(),
            auc.std(),
            precision.mean(),
            precision.std(),
            recall.mean(),
            recall.std(),
            score_time.mean()
        ])

    rows = list(clfs_par_nom.keys())

    columns = [
        "Accuracy moyenne",
        "Accuracy écart type",
        "AUC moyenne",
        "AUC écart type",
        "Précision moyenne",
        "Précision écart type",
        "Recall moyen",
        "Recall écart type",
        "Temps moyen par fold"
    ]

    df = pd.DataFrame(result_from_all_model, columns=columns, index=rows)

    print(df)

    best_model, score_final_precision_max = (
        max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1]))

    return best_model, score_final_precision_max


def run_classifiers_cv_clfs_rgs(clfs_par_nom, X, Y):
    cv_scores_par_nom_model = {}
    result_from_all_model = []
    nombre_fold = 10
    kf = KFold(n_splits=nombre_fold, shuffle=True, random_state=0)

    # Détecter le type de modèle à partir du premier modèle
    first_clf = next(iter(clfs_par_nom.values()))
    is_regression = is_regressor(first_clf)

    if is_regression:
        scoring = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "r2": "r2",
            "rmse": make_scorer(mean_absolute_error, greater_is_better=False),  # remplacé ci-dessous
        }
        scoring = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "r2": "r2",
            "rmse": make_scorer(
                lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
                greater_is_better=False
            ),
        }
        columns = [
            "MAE moyenne", "MAE écart type",
            "RMSE moyenne", "RMSE écart type",
            "R² moyen", "R² écart type",
            "Temps moyen par fold"
        ]
    else:
        scoring = {
            "accuracy": "accuracy",
            "roc_auc": make_scorer(
                roc_auc_score,
                multi_class="ovr",
                average="macro",
                needs_proba=True
            ),
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
        }
        columns = [
            "Accuracy moyenne", "Accuracy écart type",
            "AUC moyenne", "AUC écart type",
            "Précision moyenne", "Précision écart type",
            "Recall moyen", "Recall écart type",
            "Temps moyen par fold"
        ]

    for nom_model, clf in clfs_par_nom.items():
        resultats = cross_validate(clf, X, Y, cv=kf, scoring=scoring)
        score_time = resultats["score_time"]

        if is_regression:
            mae = np.abs(resultats["test_mae"])
            rmse = np.abs(resultats["test_rmse"])
            r2 = resultats["test_r2"]

            cv_scores_par_nom_model[nom_model] = r2.mean()

            result_from_all_model.append([
                mae.mean(), mae.std(),
                rmse.mean(), rmse.std(),
                r2.mean(), r2.std(),
                score_time.mean()
            ])
        else:
            accuracy = resultats["test_accuracy"]
            auc = resultats["test_roc_auc"]
            precision = resultats["test_precision"]
            recall = resultats["test_recall"]

            cv_scores_par_nom_model[nom_model] = np.mean(accuracy)

            result_from_all_model.append([
                accuracy.mean(), accuracy.std(),
                auc.mean(), auc.std(),
                precision.mean(), precision.std(),
                recall.mean(), recall.std(),
                score_time.mean()
            ])

    rows = list(clfs_par_nom.keys())
    df = pd.DataFrame(result_from_all_model, columns=columns, index=rows)
    print(df)

    best_model, score_final_max = max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1])

    return best_model, score_final_max
